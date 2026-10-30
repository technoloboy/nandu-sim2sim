"""Independent mouse window for the frozen C7 runtime; commands are unfiltered requests."""
import math

HEIGHT = ((.78, 1.04), (.68, 1.12), (.58, 1.22), (.48, 1.32), (.4, 1.4))
YAW = (.20, .50, .85, 1.20, 1.57)
VX = (.08, .20, .40, .70, 1.)


def pointer_command(dx, dy, radius, stage, height):
    if radius <= 0 or stage not in range(5) or not all(map(math.isfinite, (dx, dy, radius, height))):
        raise ValueError("Invalid mouse command")
    norm = max(radius, math.hypot(dx, dy))
    return [-dx / norm * YAW[stage], -dy / norm * VX[stage], height]


class MousePanel:
    def __init__(self, version, stage, height):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(f"NANDU1 C7-P-1-{version} — MuJoCo")
        self.root.configure(bg="#101c2e")
        self.stage, self.closed, self.reset_requested = stage, False, False
        self.axes = (0., 0.)
        tk.Label(self.root, text=f"{version} · Stage {stage} · model_30000", bg="#101c2e",
                 fg="white", font=("Sans", 15)).pack(pady=10)
        self.canvas = tk.Canvas(self.root, width=360, height=300, bg="#101c2e", highlightthickness=0)
        self.canvas.pack()
        self.center, self.radius = (180., 150.), 110.
        self.canvas.create_oval(70, 40, 290, 260, outline="#829bb6", width=2)
        self.canvas.create_line(70, 150, 290, 150, fill="#829bb6")
        self.canvas.create_line(180, 40, 180, 260, fill="#829bb6")
        for x, y, text in ((180, 20, "+vx"), (180, 280, "−vx"), (40, 150, "+yaw"), (322, 150, "−yaw")):
            self.canvas.create_text(x, y, text=text, fill="white")
        self.dot = self.canvas.create_oval(168, 138, 192, 162, fill="#32b4e1", outline="white")
        self.canvas.bind("<Button-1>", self._drag)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.root.bind("<ButtonRelease-1>", lambda e: self.stop())
        self.root.bind("<FocusOut>", self._focus_out)
        self.root.bind("<space>", lambda e: self.stop())
        lo, hi = HEIGHT[stage]
        self.height = tk.DoubleVar(value=min(hi, max(lo, height)))
        tk.Label(self.root, text="TCP absolute height (m)", bg="#101c2e", fg="white").pack()
        tk.Scale(self.root, from_=lo, to=hi, resolution=.001, orient="horizontal",
                 variable=self.height, length=410, bg="#101c2e", fg="white",
                 highlightthickness=0).pack(padx=10)
        self.status = tk.StringVar(value="Drag circle: yaw/vx. Release: zero yaw/vx, retain height.")
        tk.Label(self.root, textvariable=self.status, justify="left", bg="#101c2e", fg="#a8d9c7",
                 font=("Monospace", 10), width=55, height=7).pack()
        tk.Button(self.root, text="STOP / HOLD HEIGHT (Space)", command=self.stop).pack(fill="x", padx=12)
        tk.Button(self.root, text="RESET ROBOT", command=self._request_reset).pack(fill="x", padx=12, pady=8)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _drag(self, event):
        dx, dy = event.x-self.center[0], event.y-self.center[1]
        norm = max(1., math.hypot(dx, dy)/self.radius)
        self.axes = dx/norm, dy/norm
        x, y = self.center[0]+self.axes[0], self.center[1]+self.axes[1]
        self.canvas.coords(self.dot, x-12, y-12, x+12, y+12)

    def _focus_out(self, event):
        # Switching between this window's widgets must not interrupt a drag.
        self.root.after_idle(lambda: self.stop() if self.root.focus_displayof() is None else None)

    def stop(self):
        self.axes = 0., 0.
        self.canvas.coords(self.dot, 168, 138, 192, 162)

    def _request_reset(self):
        self.reset_requested = True
        self.stop()

    def poll(self):
        if self.closed:
            return None
        self.root.update()
        if self.closed:
            return None
        return pointer_command(*self.axes, self.radius, self.stage, self.height.get())

    def update(self, row, tilt):
        r, c = row["request"], row["command"]
        self.status.set(
            "                 yaw       vx      height\n"
            f"Request       {r[0]:+7.3f} {r[1]:+7.3f} {r[2]:7.3f}\n"
            f"Applied       {c[0]:+7.3f} {c[1]:+7.3f} {c[2]:7.3f}\n"
            f"MuJoCo truth  {row['yaw_rate']:+7.3f} {row['tcp_velocity_body'][0]:+7.3f} {row['tcp'][2]:7.3f}\n"
            f"up3 tilt {tilt:.2f} deg | time {row['time']:.2f} s")

    def reset(self, height):
        self.reset_requested = False
        self.stop()
        self.height.set(min(HEIGHT[self.stage][1], max(HEIGHT[self.stage][0], height)))

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
