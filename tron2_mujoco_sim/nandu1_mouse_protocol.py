"""Local mouse protocol and safety state. No ROS, Tk, or policy imports."""
import json
import math
import socket
import threading

SCHEMA = 'nandu.mouse-ipc/v1'
MAX_LINE = 4096


def frame(kind, session, task, **fields):
    return dict(schema=SCHEMA, kind=kind, session=session, task=task, **fields)


def vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError('Invalid vector size')
    try:
        valid = all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                    for v in value)
        result = [float(v) for v in value] if valid else None
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('Expected finite numeric vector')
    return result


def clamp(value, low, high):
    return max(low, min(high, value))


def command_interface(task):
    import nandu1_contract as contract
    from nandu1_adaptive_contract import deployment_profile
    profile = deployment_profile(task)
    family = ('heading' if profile['actor_velocity_frame'] == 'heading' else
              'tcp' if profile['velocity_reference'] == 'fixed_tcp' else 'body')
    topic = ('/nandu1/command' if profile['task_id'] == contract.TASK else
             '/nandu1/' + profile['task_id'].split('Flat-')[1].replace('-', '_') + '/command')
    return family, topic


def assert_external_available(owner, publishers):
    if isinstance(owner, dict) and owner.get('node') and owner.get('topic'):
        if any(topic == owner['topic'] and owner['node'] in nodes for topic, nodes in publishers):
            raise RuntimeError('Active mouse command source: close its session before keyboard/steps/lifecycle')


class ControllerStatus:
    """Sensor/controller-only status; never consumes evaluation truth."""
    def __init__(self, family, initial_height):
        self.size = {'body': 116, 'tcp': 121, 'heading': 131}[family]
        self.initial_height = initial_height
        self._track = self._reset = None
        self._diagnostic = ('Waiting for controller', -math.inf)
        self.lock = threading.RLock()

    def tracking(self, data, now):
        data = vector(list(data), self.size)
        if data[1] < 0 or data[1] != int(data[1]):
            raise ValueError('Invalid tracking epoch')
        with self.lock:
            self._track = (int(data[1]), data[2:5], now, data[0])

    def reset(self, data):
        data = vector(list(data), 12)
        if any(v < 0 or v != int(v) for v in data[1:3]):
            raise ValueError('Invalid controller reset key')
        with self.lock:
            self._reset = (int(data[1]), int(data[2]), data[11], data[0])

    def diagnostic(self, mode, now):
        with self.lock:
            self._diagnostic = (mode, now)

    def snapshot(self, now):
        with self.lock:
            mode, received = self._diagnostic
            key = [self._track[0], self._reset[1]] if self._track and self._reset else None
            fresh = (self._track is not None and self._reset is not None and
                     self._track[0] == self._reset[0] and self._track[3] >= self._reset[3] and
                     now - self._track[2] <= .5 and
                     now - received <= .5)
            return dict(ready=bool(fresh and mode == 'POLICY'), key=key,
                        applied=list(self._track[1]) if self._track else [0., 0., self.initial_height],
                        initial_height=self._reset[2] if self._reset else self.initial_height,
                        mode=mode if fresh else ('FAULT: ' + mode if mode != 'POLICY' else 'Stale controller state'))


class ManualState:
    def __init__(self, lower, upper, height):
        self.lower, self.upper = vector(lower, 3), vector(upper, 3)
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError('Invalid command bounds')
        self.height = clamp(vector([height], 1)[0], self.lower[2], self.upper[2])
        self.gain = .25
        self.focused = self.ready = self.armed = self.active = False
        self.axes = [0., 0.]
        self.key = None
        self.arm_token = 0
        self.last_status = -math.inf

    def release(self):
        self.axes = [0., 0.]
        self.active = False

    def disarm(self):
        self.release()
        self.armed = False

    def focus(self, focused):
        self.focused = bool(focused)
        if not focused:
            self.disarm()

    def status(self, ready, key, applied, now, arm_token=None):
        applied = vector(applied, 3)
        if arm_token is not None and (type(arm_token) is not int or arm_token < 0):
            raise ValueError('Invalid bridge rearm token')
        if (tuple(key) != self.key or (ready and not self.ready) or
                (arm_token is not None and arm_token != self.arm_token)):
            self.disarm()
            self.height = clamp(applied[2], self.lower[2], self.upper[2])
        if arm_token is not None:
            self.arm_token = arm_token
        self.key, self.ready, self.last_status = tuple(key), bool(ready), now
        if not ready:
            self.disarm()

    def expire(self, now):
        if now - self.last_status > .5:
            self.ready = False
            self.disarm()

    def begin(self):
        if self.focused and self.ready:
            self.active = self.armed = True

    def drag(self, dx, dy):
        if not (self.active and self.armed and self.focused and self.ready):
            return
        dx, dy = vector([dx, dy], 2)
        radius = math.hypot(dx, dy)
        if radius <= .05:
            self.axes = [0., 0.]
        else:
            magnitude = (min(radius, 1.) - .05) / .95
            self.axes = [-dx / radius * magnitude, -dy / radius * magnitude]

    def set_gain(self, gain):
        gain = vector([gain], 1)[0]
        if not 0 < gain <= 1:
            raise ValueError('Gain must be in (0,1]')
        self.gain = gain

    def set_height(self, height):
        if self.focused and self.ready:
            self.height = clamp(vector([height], 1)[0], self.lower[2], self.upper[2])
            self.armed = True

    def values(self):
        axes = self.axes if self.ready and self.focused and self.armed else [0., 0.]
        return [axis * self.gain * (hi if axis >= 0 else -lo)
                for axis, lo, hi in zip(axes, self.lower, self.upper)] + [self.height]

    def command(self, session, task, seq, reset=False, closing=False):
        return frame('command', session, task, seq=seq, key=list(self.key),
                     arm_token=self.arm_token, closing=closing,
                     enabled=bool(self.ready and self.focused and self.armed),
                     axes=[a * self.gain for a in self.axes], height=self.height, reset=reset)


class CommandGate:
    """Reject stale/old-session commands; only neutral input clears rearm latch."""
    def __init__(self, session, task, lower, upper, height):
        self.session, self.task = session, task
        self.lower, self.upper = vector(lower, 3), vector(upper, 3)
        self.request = [0., 0., height]
        self.seq, self.key = -1, None
        self.ready = False
        self.needs_release = True
        self.arm_token = 0
        self.closing = False
        self.received = -math.inf
        self.reason = 'Waiting for controller'

    def neutral(self, reason, renew=False):
        # A server-issued challenge prevents buffered pre-timeout neutral/drag
        # pairs from re-enabling motion after a bridge scheduling pause.
        if renew or not self.needs_release:
            self.arm_token += 1
        self.request[:2] = [0., 0.]
        self.needs_release = True
        self.reason = reason

    def sync(self, ready, key, initial_height):
        key = tuple(key) if key is not None else None
        if key != self.key:
            self.neutral('Controller reset: release and drag again', renew=True)
            self.request[2] = clamp(initial_height, self.lower[2], self.upper[2])
            self.received = -math.inf
        if self.ready and not ready:
            self.neutral('Controller not ready')
        self.ready, self.key = bool(ready), key

    def receive(self, packet, now):
        self.values(now)  # Check age before incoming frames can refresh it.
        try:
            if not isinstance(packet, dict) or any(packet.get(k) != v for k, v in (
                ('schema', SCHEMA), ('kind', 'command'), ('session', self.session), ('task', self.task))):
                raise ValueError('Invalid mouse session/task/schema')
            seq = packet.get('seq')
            if isinstance(seq, bool) or not isinstance(seq, int) or seq <= self.seq:
                raise ValueError('Mouse sequence is not increasing')
            if vector(packet.get('key'), 2) != list(self.key or ()):
                raise ValueError('Mouse epoch/generation mismatch')
            enabled, reset = packet.get('enabled'), packet.get('reset')
            closing = packet.get('closing', False)
            if type(enabled) is not bool or type(reset) is not bool or type(closing) is not bool:
                raise ValueError('Invalid mouse enable/reset flag')
            axes = vector(packet.get('axes'), 2)
            height = vector([packet.get('height')], 1)[0]
            if any(abs(a) > 1 for a in axes) or not self.lower[2] <= height <= self.upper[2]:
                raise ValueError('Mouse command outside limits')
            if closing:
                self.closing = True
                self.seq = seq
                self.neutral('Mouse window closing')
                return False
            if self.closing:
                return False
            token = packet.get('arm_token')
            if type(token) is not int or token < 0:
                raise ValueError('Invalid mouse rearm token')
            if token != self.arm_token:
                self.reason = 'Expired rearm token: release and drag again'
                return False
            self.seq, self.received = seq, now
            if reset:
                self.neutral('Explicit controller RESET requested')
                return True
            if not enabled:
                self.request[:2] = [0., 0.]
                self.needs_release = False
                self.reason = 'Stopped / height held'
            elif self.ready and not self.needs_release:
                self.request = [a * (hi if a >= 0 else -lo)
                                for a, lo, hi in zip(axes, self.lower, self.upper)] + [height]
                self.reason = 'Mouse control active'
            return False
        except (ValueError, TypeError, KeyError) as exc:
            self.neutral('Rejected input: ' + str(exc))
            raise ValueError(str(exc)) from None

    def values(self, now):
        if now - self.received > .25 and not self.needs_release:
            self.neutral('Mouse input timeout: release and drag again')
        if not self.ready:
            self.request[:2] = [0., 0.]
        return list(self.request)


class JsonChannel:
    """Bounded JSON Lines; refuse a new send while a partial frame is pending."""
    def __init__(self, sock):
        self.sock = sock
        sock.setblocking(False)
        self.incoming = bytearray()
        self.outgoing = b''

    def flush(self):
        if self.outgoing:
            try:
                count = self.sock.send(self.outgoing)
            except BlockingIOError:
                return
            if count == 0:
                raise ConnectionError('Mouse IPC closed')
            self.outgoing = self.outgoing[count:]

    def send(self, packet):
        payload = (json.dumps(packet, allow_nan=False, separators=(',', ':')) + '\n').encode()
        if len(payload) > MAX_LINE:
            raise ValueError('Mouse IPC frame too large')
        self.flush()
        if self.outgoing:
            return False
        self.outgoing = payload
        self.flush()
        return True

    def receive(self):
        try:
            data = self.sock.recv(MAX_LINE * 8)
        except BlockingIOError:
            data = None
        if data == b'':
            raise ConnectionError('Mouse IPC peer closed')
        if data:
            self.incoming.extend(data)
        result = []
        while b'\n' in self.incoming:
            line, _, rest = self.incoming.partition(b'\n')
            if len(line) + 1 > MAX_LINE:
                raise ValueError('Mouse IPC frame too large')
            self.incoming = bytearray(rest)
            def reject_constant(_):
                raise ValueError('Nonfinite mouse JSON')
            try:
                parsed = json.loads(line, parse_constant=reject_constant)
            except RecursionError:
                raise ValueError('Mouse IPC JSON nesting too deep') from None
            if not isinstance(parsed, dict):
                raise ValueError('Expected mouse JSON object')
            result.append(parsed)
        if len(self.incoming) >= MAX_LINE:
            raise ValueError('Mouse IPC unterminated frame too large')
        self.flush()
        return result
