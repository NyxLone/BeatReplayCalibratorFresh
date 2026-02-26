import struct
import io

class Bsor:
    def __init__(self, stream: io.BufferedIOBase):
        self.stream = stream
        self.notes = []
        self._read_header()
        self._read_notes()

    def _read_header(self):
        self.stream.read(4)  # magic
        self.stream.read(4)  # version

    def _read_notes(self):
        try:
            while True:
                t = struct.unpack("<f", self.stream.read(4))[0]
                saber = struct.unpack("<b", self.stream.read(1))[0]
                pre = struct.unpack("<f", self.stream.read(4))[0]
                post = struct.unpack("<f", self.stream.read(4))[0]
                dist = struct.unpack("<f", self.stream.read(4))[0]

                self.notes.append({
                    "time": t,
                    "saberType": saber,
                    "preSwing": pre,
                    "postSwing": post,
                    "cutDistanceToCenter": dist,
                })
        except:
            pass

def make_bsor(stream):
    return Bsor(stream)
