#!/usr/bin/env python3
"""Generate a gentle, royalty-free elevator-music bed sized to the storyboard length.

Soft sine-pad chord progression (I-vi-IV-V) with slow attack, low volume. No samples,
no copyright. Writes assets/music.wav.
"""
import json, numpy as np, wave, struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
SB = json.load(open(HERE / "storyboard.json"))
DUR = sum(s["dur"] for s in SB["scenes"])
SR = 44100

def note(freq, t):
    # sine + soft octave, gentle vibrato
    v = 0.5*np.sin(2*np.pi*freq*t) + 0.2*np.sin(2*np.pi*2*freq*t)
    v *= (1 + 0.004*np.sin(2*np.pi*5*t))
    return v

def chord(freqs, dur):
    t = np.linspace(0, dur, int(SR*dur), endpoint=False)
    x = sum(note(f, t) for f in freqs) / len(freqs)
    # soft attack/release envelope
    env = np.ones_like(t); a = int(SR*0.4)
    env[:a] = np.linspace(0,1,a); env[-a:] = np.linspace(1,0,a)
    return x*env

# C major: I(C) vi(Am) IV(F) V(G), 4s each, looped to length
prog = [[261.63,329.63,392.00],[220.00,261.63,329.63],
        [174.61,220.00,261.63],[196.00,246.94,293.66]]
bar = 4.0
out = []
tacc = 0.0
i = 0
while tacc < DUR:
    out.append(chord(prog[i % 4], bar)); tacc += bar; i += 1
audio = np.concatenate(out)[:int(SR*DUR)]
audio *= 0.12 / (np.max(np.abs(audio))+1e-9)   # quiet bed
# gentle low-pass (moving average) for warmth
k = 30; audio = np.convolve(audio, np.ones(k)/k, mode="same")
pcm = (audio*32767).astype(np.int16)
Path(HERE/"assets").mkdir(exist_ok=True)
with wave.open(str(HERE/"assets"/"music.wav"),"w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"music.wav written, {DUR}s")
