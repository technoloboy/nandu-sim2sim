"""Test-only driver: inject real Tk events into the production mouse window."""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nandu1_mouse_gui as gui


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--container')
    args,rest=parser.parse_known_args()
    original=gui.MousePanel.__init__
    results=[];events=[];start=time.monotonic()

    def initialized(self,*a,**kw):
        original(self,*a,**kw)
        state=self.state;root=self.root
        phase=-1;began=0.;old_generation=None;last_state=None
        guard_result=[]
        def check_source_guard():
            command=['docker','exec',args.container,'bash','-lc',
                     'source /ws/devel/setup.bash && rosrun tron2_controllers nandu1_command.py --mode steps --task '+a[2]]
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=5)
                guard_result.append(dict(returncode=result.returncode,output=result.stdout+result.stderr))
            except Exception as exc:
                guard_result.append(dict(error=str(exc)))
            args.report.with_suffix('.guard.json').write_text(json.dumps(guard_result,indent=2))
        def drive():
            nonlocal phase,began,old_generation,last_state
            now=time.monotonic()
            snapshot=(state.ready,state.focused,state.armed,state.arm_token,self.status_var.get())
            if snapshot!=last_state:
                events.append(dict(t=now-start,phase=phase,state=snapshot))
                last_state=snapshot
            if now-start>25:
                results.append(dict(error='Controller/GUI readiness timeout',mode=self.status_var.get()))
                finish();return
            if phase==-1:
                if state.ready:
                    root.focus_force();root.update_idletasks();state.focus(True)
                    phase=0;began=now
                    self.canvas.event_generate('<ButtonPress-1>',x=180,y=180)
                    self.canvas.event_generate('<B1-Motion>',x=180,y=140)
            elif phase in (0,1,2,3) and now-began>.6:
                results.append(dict(phase=phase,target=state.values(),applied=self.applied_var.get(),status=self.status_var.get()))
                if phase==0:
                    # Inspect the real widget and model, not telemetry's copy
                    # of the manifest. No need to command unsafe extremes.
                    results[-1]['ui']=dict(
                        stage_label=root.winfo_children()[0].winfo_children()[1].cget('text'),
                        lower=list(state.lower),upper=list(state.upper),
                        slider_from=float(self.height_scale.cget('from')),
                        slider_to=float(self.height_scale.cget('to')))
                self.canvas.event_generate('<ButtonRelease-1>',x=180,y=180)
                phase+=1;began=now
                if phase in (1,2,3):
                    # MuJoCo can take focus when its delayed viewer appears.
                    # Each test gesture explicitly selects OUR command window.
                    root.focus_force();root.update_idletasks();state.focus(True)
                    x,y={1:(180,220),2:(140,180),3:(220,180)}[phase]
                    self.canvas.event_generate('<ButtonPress-1>',x=180,y=180)
                    self.canvas.event_generate('<B1-Motion>',x=x,y=y)
                else:
                    root.focus_force();root.update_idletasks();state.focus(True)
                    self.height_var.set(state.height+.02);self.set_height(str(state.height+.02))
                    if args.container:
                        threading.Thread(target=check_source_guard,daemon=True).start()
            elif phase==4 and now-began>.6:
                results.append(dict(phase=phase,target=state.values(),applied=self.applied_var.get()))
                old_generation=state.key[1]
                self.reset_button.invoke();phase=5;began=now
            elif phase==5 and state.ready and state.key[1]>old_generation:
                results.append(dict(phase=phase,key=list(state.key),target=state.values(),status=self.status_var.get()))
                phase=6;began=now
            elif phase==6 and now-began>.3 and (not args.container or guard_result):
                finish();return
            root.after(20,drive)
        def finish():
            args.report.write_text(json.dumps(results,indent=2))
            args.report.with_suffix('.events.json').write_text(json.dumps(events,indent=2))
            root.after_idle(lambda:root.event_generate('<<ProbeFinished>>'))
            # Use the production WM close callback, which sends neutral first.
            root.after(20,lambda:root.tk.call(root.protocol('WM_DELETE_WINDOW')))
        root.after(20,drive)
    gui.MousePanel.__init__=initialized
    sys.argv=[sys.argv[0],*rest]
    return gui.main()


if __name__=='__main__':raise SystemExit(main())
