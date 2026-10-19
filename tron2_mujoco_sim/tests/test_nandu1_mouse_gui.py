"""Real Tk events against our own panel; skip only when no display exists."""
import importlib
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_real_tk_drag_release_height_and_focus_loss():
    if not os.environ.get('DISPLAY'):
        pytest.skip('No display for actual Tk event test')
    import tkinter as tk
    gui = importlib.import_module('nandu1_mouse_gui')
    from nandu1_mouse_protocol import ManualState
    root = tk.Tk()
    try:
        state = ManualState([-1.57,-1,.4],[1.57,1,1.4],.9)
        view = gui.MousePanel(root,state,'Isaac-NANDU1-Blind-Flat-v1-2','body',4)
        root.update(); root.focus_force(); root.update()
        import tkinter.font as tkfont
        title=root.winfo_children()[0].winfo_children()[0]
        assert title.cget('text')=='Isaac-NANDU1-Blind-Flat-v1-2'
        assert tkfont.Font(font=title.cget('font')).metrics('linespace')>0
        state.status(True,(0,1),[0,0,.9],1.)
        view.canvas.event_generate('<ButtonPress-1>',x=180,y=180)
        view.canvas.event_generate('<B1-Motion>',x=180,y=50)
        root.update()
        assert state.values()[1] == pytest.approx(.25)
        view.canvas.event_generate('<ButtonRelease-1>',x=180,y=50)
        root.update()
        assert state.values() == [0.,0.,.9]
        view.height_var.set(1.1); view.set_height('1.1')
        assert state.values()[2] == 1.1
        state.begin(); state.drag(0,-1)
        state.focus(False); view.paint([0,0,1.1],'POLICY','focus lost')
        assert state.values() == [0.,0.,1.1]
        assert '1.100' in view.target_var.get()
    finally:
        root.destroy()


def test_owned_gui_signal_exits_promptly_and_removes_private_socket(tmp_path):
    if not os.environ.get('DISPLAY'):
        pytest.skip('No display for actual GUI lifecycle test')
    import signal
    import subprocess
    import time
    sim=Path(__file__).resolve().parents[1]
    endpoint=tmp_path/'control.sock'
    marker=tmp_path/'inside_callback'
    arguments=['--task','Isaac-NANDU1-Blind-Flat-v1-2',
               '--bundle',str(sim.parent/'deployment_bundles/nandu1_v1_2_model_40000'),
               '--session','test-gui-lifecycle','--socket',str(endpoint)]
    # Deliver the signal INSIDE a Tk callback: Tk normally catches Python
    # callback exceptions, so catching only around mainloop is insufficient.
    code=f'''import sys,time
from pathlib import Path
sys.path.insert(0,{str(sim)!r})
import nandu1_mouse_gui as gui
original=gui.MousePanel.__init__
def initialized(self,*a,**kw):
    original(self,*a,**kw)
    def busy():
        Path({str(marker)!r}).write_text('inside callback')
        time.sleep(.8)
    self.root.after(200,busy)
gui.MousePanel.__init__=initialized
sys.argv=['nandu1_mouse_gui.py',*{arguments!r}]
raise SystemExit(gui.main())
'''
    command=[sys.executable,'-c',code]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+5
        while not marker.exists():
            if process.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('GUI did not initialize')
            time.sleep(.02)
        process.send_signal(signal.SIGINT)
        _,error=process.communicate(timeout=2)
        assert process.returncode==0,error
        assert not endpoint.exists()
    finally:
        if process.poll() is None:
            process.kill();process.communicate()
