"""Host-only Tk mouse window. Optional: importing this module never imports Tk."""
import argparse
import os
from pathlib import Path
import socket
import struct
import time
import signal

from nandu1_mouse_protocol import JsonChannel, ManualState, SCHEMA, command_interface, vector


def probe_display():
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('Mouse control needs a desktop DISPLAY; use external for headless collection')
    import tkinter as tk
    try:
        root = tk.Tk(); root.withdraw(); root.destroy()
    except tk.TclError as exc:
        raise RuntimeError('Cannot create mouse window: ' + str(exc)) from None


class MousePanel:
    def __init__(self, root, state, task, family, stage):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.state = root, state
        self.reset_pending = False
        root.title('NANDU1 | Mouse command | MuJoCo only')
        root.geometry('760x570')
        root.minsize(720, 540)
        root.configure(background='#101b2c')
        header = tk.Frame(root, bg='#101b2c'); header.pack(fill='x', padx=20, pady=(15, 8))
        tk.Label(header, text=task, bg='#101b2c', fg='#f0f5ff', font='TkDefaultFont').pack(anchor='w')
        reference = {'body':'yaw: world Z rate | vx: base Body | height: up3 world Z',
                     'tcp':'yaw: relative heading rate | vx: TCP in base Body | height: TCP world Z',
                     'heading':'yaw: relative heading rate | vx: base Heading | height: up3 world Z'}[family]
        tk.Label(header, text=f'Stage: {stage} | {reference}', bg='#101b2c', fg='#99bbdc').pack(anchor='w', pady=4)
        body = tk.Frame(root, bg='#101b2c'); body.pack(fill='both', expand=True, padx=20)
        self.canvas = tk.Canvas(body, width=360, height=360, bg='#172840', highlightthickness=0)
        self.canvas.pack(side='left', padx=(0, 35))
        self.canvas.create_oval(50, 50, 310, 310, outline='#6482a5', width=2)
        self.canvas.create_line(50, 180, 310, 180, fill='#415a79')
        self.canvas.create_line(180, 50, 180, 310, fill='#415a79')
        for x, y, text in ((180,25,'Forward +vx'),(180,337,'Backward -vx'),(42,180,'+yaw'),(324,180,'-yaw')):
            self.canvas.create_text(x, y, text=text, fill='#bdcee4')
        self.knob = self.canvas.create_oval(160,160,200,200,fill='#4ee0b8',outline='')
        self.canvas.bind('<ButtonPress-1>', self.begin)
        self.canvas.bind('<B1-Motion>', self.drag)
        root.bind('<ButtonRelease-1>', self.release, add='+')
        root.bind('<FocusIn>', lambda _: root.after_idle(self.sync_focus), add='+')
        root.bind('<FocusOut>', lambda _: root.after_idle(self.sync_focus), add='+')
        root.bind('<Escape>', lambda _: self.stop())
        right = tk.Frame(body, bg='#101b2c'); right.pack(side='left', fill='both', expand=True)
        tk.Label(right, text='Absolute height (m)', bg='#101b2c', fg='#f0f5ff').pack()
        self.height_var = tk.DoubleVar(value=state.height)
        self.height_scale = tk.Scale(right, from_=state.upper[2], to=state.lower[2], resolution=.01,
            variable=self.height_var, command=self.set_height, orient='vertical', length=255,
            bg='#172840', fg='#e7f2ff', troughcolor='#263d59', highlightthickness=0, width=28)
        self.height_scale.pack(pady=8)
        gains = tk.Frame(right, bg='#101b2c'); gains.pack()
        tk.Label(gains,text='Joystick gain', bg='#101b2c',fg='#bdcee4').pack(side='left')
        self.gain_var = tk.StringVar(value='0.25')
        gain = ttk.Combobox(gains, textvariable=self.gain_var, values=('0.1','0.25','0.5','1.0'),state='readonly',width=5)
        gain.pack(side='left',padx=5)
        gain.bind('<<ComboboxSelected>>', lambda _: self.state.set_gain(float(self.gain_var.get())))
        bottom = tk.Frame(root,bg='#101b2c'); bottom.pack(fill='x',padx=20,pady=12)
        self.status_var = tk.StringVar(value='Waiting for ROS controller')
        self.target_var = tk.StringVar(); self.applied_var = tk.StringVar()
        for variable,color in ((self.status_var,'#4ee0b8'),(self.target_var,'#ffd58a'),(self.applied_var,'#a1c8ff')):
            tk.Label(bottom,textvariable=variable,bg='#101b2c',fg=color,anchor='w').pack(fill='x')
        actions = tk.Frame(bottom,bg='#101b2c'); actions.pack(fill='x',pady=(8,0))
        ttk.Button(actions,text='STOP / hold height',command=self.stop).pack(side='left')
        self.reset_button = ttk.Button(actions,text='Reset controller (explicit)',command=self.request_reset)
        self.reset_button.pack(side='left',padx=10)
        tk.Label(actions,text='Release / focus loss = zero yaw & vx',bg='#101b2c',fg='#91a7c1').pack(side='right')
        self.paint([0,0,state.height],'Waiting','Drag only after POLICY is ready')

    def sync_focus(self):
        widget = self.root.focus_get()
        self.state.focus(widget is not None and widget.winfo_toplevel() == self.root)

    def begin(self, event):
        self.sync_focus(); self.state.begin(); self.drag(event)

    def drag(self, event):
        self.state.drag((event.x-180)/130., (event.y-180)/130.)
        self.draw_knob()

    def release(self, _event=None):
        self.state.release(); self.draw_knob()

    def stop(self):
        self.state.disarm(); self.draw_knob()

    def request_reset(self):
        self.stop(); self.reset_pending = True

    def set_height(self, value):
        self.state.set_height(float(value))

    def draw_knob(self):
        yaw, vx = self.state.axes
        x, y = 180-yaw*130, 180-vx*130
        self.canvas.coords(self.knob,x-20,y-20,x+20,y+20)

    def paint(self, applied, mode, reason):
        self.draw_knob()
        if abs(self.height_var.get()-self.state.height) > .005:
            self.height_var.set(self.state.height)
        self.status_var.set(f'{mode} | {reason}')
        self.target_var.set('Requested: yaw={:+.3f} rad/s | vx={:+.3f} m/s | height={:.3f} m'.format(*self.state.values()))
        self.applied_var.set('Applied:     yaw={:+.3f} rad/s | vx={:+.3f} m/s | height={:.3f} m'.format(*applied))
        self.height_scale.configure(state='normal' if self.state.ready else 'disabled')
        self.reset_button.configure(state='normal' if self.state.key is not None else 'disabled')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args()
    from nandu1_bundle import load_bundle
    manifest, _ = load_bundle(args.bundle, args.task)
    task = manifest['task']; canonical = task['task_id']
    family, _ = command_interface(canonical)
    state = ManualState(task['command_min'],task['command_max'],task['initial_up3_height'])
    import tkinter as tk
    root = tk.Tk()
    view = MousePanel(root,state,canonical,family,task.get('evaluation_stage','legacy'))
    listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(str(args.socket)); os.chmod(args.socket,0o600)
    listener.listen(1); listener.setblocking(False)
    channel = None; sequence = 0; last_status_seq = -1
    closing = False
    applied = [0.,0.,state.height]; mode = 'Waiting'; reason = 'Waiting for ROS bridge'

    def disconnect(why):
        nonlocal channel, mode, reason, last_status_seq
        if channel is not None: channel.sock.close()
        channel = None; last_status_seq = -1
        state.ready = False; state.disarm()
        mode, reason = 'Disconnected', why

    def tick():
        nonlocal channel, sequence, last_status_seq, applied, mode, reason
        now = time.monotonic()
        try:
            if channel is None:
                try:
                    sock, _ = listener.accept()
                    _, uid, _ = struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    if uid not in (0,os.getuid()):
                        sock.close(); raise ValueError('IPC peer is not the local session owner')
                    channel = JsonChannel(sock); state.disarm()
                except BlockingIOError:
                    pass
            if channel is not None:
                for packet in channel.receive():
                    if any(packet.get(k) != value for k,value in (
                        ('schema',SCHEMA),('kind','status'),('session',args.session),('task',canonical))):
                        raise ValueError('Invalid bridge identity')
                    seq = packet.get('seq')
                    if type(seq) is not int or seq <= last_status_seq:
                        raise ValueError('Invalid bridge sequence')
                    if packet.get('lower') != state.lower or packet.get('upper') != state.upper:
                        raise ValueError('Bridge command bounds differ from bundle')
                    last_status_seq = seq
                    applied = vector(packet['applied'],3)
                    mode, reason = packet['mode'], packet['reason']
                    if type(packet['ready']) is not bool:
                        raise ValueError('Invalid bridge ready flag')
                    key = packet['key']
                    if key is not None:
                        if not isinstance(key,list) or len(key)!=2 or any(type(k) is not int or k<0 for k in key):
                            raise ValueError('Invalid bridge reset key')
                        state.status(packet['ready'] and not closing,key,applied,now,
                                     arm_token=packet['arm_token'])
                state.expire(now)
                if state.key is not None:
                    packet = state.command(args.session,canonical,sequence,
                                           view.reset_pending and not closing,closing=closing)
                    if channel.send(packet):
                        sequence += 1; view.reset_pending = False
            view.paint(applied,mode,reason)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            disconnect(str(exc)); view.paint(applied,mode,reason)
        root.after(20,tick)

    def close():
        nonlocal closing, sequence
        if closing:
            return
        closing = True
        view.reset_pending = False
        state.disarm()
        if channel is not None and state.key is not None:
            try:
                if channel.send(state.command(args.session,canonical,sequence,closing=True)):
                    sequence += 1
            except (OSError,ValueError): pass
        # Allow the ROS bridge to switch to PASSIVE before the supervisor stops
        # physics. Tick keeps retransmitting neutral closing intents meanwhile.
        root.after(120,root.destroy)

    root.protocol('WM_DELETE_WINDOW',close)
    # Tk catches exceptions from callbacks, including KeyboardInterrupt. An
    # explicit handler schedules the SAME safe close path instead of relying
    # on an exception escaping root.mainloop(). Only this GUI process is changed.
    previous_signals = {number:signal.signal(number,lambda *_:root.after_idle(close))
                        for number in (signal.SIGINT,signal.SIGTERM)}
    root.after(20,tick)
    try:
        root.mainloop()
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        for number,handler in previous_signals.items():
            signal.signal(number,handler)
        if channel is not None:
            if not closing and state.key is not None:
                state.disarm()
                try:
                    channel.send(state.command(args.session,canonical,sequence,closing=True))
                    deadline = time.monotonic()+.12
                    while time.monotonic()<deadline:
                        channel.flush(); time.sleep(.02)
                except (OSError,ValueError): pass
            channel.sock.close()
        listener.close()
        args.socket.unlink(missing_ok=True)
        try: root.destroy()
        except tk.TclError: pass


if __name__ == '__main__':
    raise SystemExit(main())
