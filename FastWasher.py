#!/usr/bin/env python3
"""
FastWasher - Serum FXP to FastTracker 2 XI Converter
"""

import os
import sys
import time
import shutil
import struct
import re
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import dawdreamer as daw
    import numpy as np
    from scipy.io import wavfile
except ImportError as e:
    print(f"Missing library: {e}")
    print("\nInstall with:")
    print("pip install dawdreamer numpy scipy")
    sys.exit(1)

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
NOTE_VELOCITY = 100
NOTE_DURATION = 4.0
RENDER_TAIL = 3.0
MAX_TAIL_EXTENSION = 10.0
SILENCE_THRESHOLD_DB = -80
SILENCE_CHECK_WINDOW = 0.1

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

NOTE_RE = re.compile(r'.*_([A-G]#?)(\d+)_(\d+)$')

ASCII_HEADER = r"""
·▄▄▄ ▄▄▄· .▄▄ · ▄▄▄▄▄▄▄▌ ▐ ▄▌ ▄▄▄· .▄▄ ·  ▄ .▄▄▄▄ .▄▄▄  
▐▄▄·▐█ ▀█ ▐█ ▀. •██  ██· █▌▐█▐█ ▀█ ▐█ ▀. ██▪▐█▀▄.▀·▀▄ █·
██▪ ▄█▀▀█ ▄▀▀▀█▄ ▐█.▪██▪▐█▐▐▌▄█▀▀█ ▄▀▀▀█▄██▀▐█▐▀▀▪▄▐▀▀▄ 
██▌.▐█ ▪▐▌▐█▄▪▐█ ▐█▌·▐█▌██▐█▌▐█ ▪▐▌▐█▄▪▐███▌▐▀▐█▄▄▌▐█•█▌
▀▀▀  ▀  ▀  ▀▀▀▀  ▀▀▀  ▀▀▀▀ ▀▪ ▀  ▀  ▀▀▀▀ ▀▀▀ · ▀▀▀ .▀  ▀"""

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_header():
    """Display the ASCII header"""
    clear_screen()
    print(ASCII_HEADER)

def show_section(current, total, title, clear=True):
    """Display a section header"""
    if clear:
        clear_screen()
    print(f"\n{'═'*70}")
    print(f"  STEP {current}/{total}: {title}")
    print(f"{'═'*70}\n")

def progress_bar(current, total, bar_length=40, prefix="Progress", extra=""):
    """Display a progress bar"""
    percent = float(current) / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent_str = f"{percent * 100:.1f}%"
    return f"\r{prefix}: [{bar}] {percent_str} ({current}/{total}) {extra}"

def format_time(seconds):
    """Format seconds into readable time string"""
    return str(timedelta(seconds=int(seconds)))

def db_to_linear(db):
    """Convert dB to linear amplitude"""
    return 10 ** (db / 20)

def linear_to_db(linear):
    """Convert linear amplitude to dB"""
    if linear <= 0:
        return -np.inf
    return 20 * np.log10(linear)

def check_file_access(path):
    """Check if we can access a file"""
    try:
        with open(path, 'rb') as f:
            f.read(4)
        return True
    except:
        return False

def get_file_architecture(path):
    """Try to determine if a DLL is 32-bit or 64-bit"""
    try:
        with open(path, 'rb') as f:
            dos_header = f.read(64)
            if dos_header[:2] != b'MZ':
                return "Unknown"
            
            pe_offset = int.from_bytes(dos_header[60:64], 'little')
            f.seek(pe_offset)
            pe_sig = f.read(4)
            if pe_sig != b'PE\x00\x00':
                return "Unknown"
            
            machine_type = f.read(2)
            machine = int.from_bytes(machine_type, 'little')
            
            if machine == 0x14c:
                return "32-bit"
            elif machine == 0x8664:
                return "64-bit"
            else:
                return f"Unknown (0x{machine:x})"
    except:
        return "Unknown"

def find_all_serum_versions():
    """Find all Serum installations"""
    found = []
    
    known_paths = [
        r"C:\Program Files\VSTPlugins\Xfer\Serum\Serum_x64.dll",
        r"C:\Program Files\VSTPlugins\Serum_x64.dll",
        r"C:\Program Files\VSTPlugins\Serum.dll",
        r"C:\Program Files\Steinberg\VSTPlugins\Serum.dll",
        r"C:\Program Files\Steinberg\VSTPlugins\Serum_x64.dll",
        r"C:\Program Files\Common Files\VST2\Serum.dll",
        r"C:\Program Files\Common Files\VST2\Serum_x64.dll",
        r"C:\Program Files (x86)\VSTPlugins\Serum.dll",
        r"C:\Program Files (x86)\Steinberg\VSTPlugins\Serum.dll",
        r"C:\Program Files\Common Files\VST3\Serum.vst3",
    ]
    
    for path in known_paths:
        if Path(path).exists():
            arch = get_file_architecture(path) if path.endswith('.dll') else "VST3"
            accessible = check_file_access(path)
            found.append({
                'path': path,
                'type': 'VST3' if path.endswith('.vst3') else 'VST2',
                'arch': arch,
                'accessible': accessible
            })
    
    return found

def test_vst_load(vst_path):
    """Test if DawDreamer can load a VST"""
    try:
        engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
        synth = engine.make_plugin_processor("serum_test", vst_path)
        return True
    except Exception as e:
        print(f"Error loading VST: {e}")
        return False

def get_note_name(midi_num):
    """Convert MIDI number to note name"""
    octave = (midi_num - 12) // 12
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"

def get_midi_range(start="C0", end="G9"):
    """Get all MIDI notes in range"""
    def note_to_midi(note_str):
        if '#' in note_str:
            note = note_str[:2]
            octave = int(note_str[2:]) if len(note_str) > 2 else 0
        else:
            note = note_str[0]
            octave = int(note_str[1:]) if len(note_str) > 1 else 0
        
        try:
            note_index = NOTE_NAMES.index(note.upper())
        except ValueError:
            raise ValueError(f"Invalid note: {note_str}")
        
        return 12 + octave * 12 + note_index
    
    start_midi = note_to_midi(start)
    end_midi = note_to_midi(end)
    actual_end = min(end_midi, 127)
    
    return [(n, get_note_name(n)) for n in range(start_midi, actual_end + 1)]

def get_optimal_16_notes(start="C0", end="G9"):
    """Get exactly 16 evenly distributed MIDI notes for optimal XI sampling"""
    def note_to_midi(note_str):
        if '#' in note_str:
            note = note_str[:2]
            octave = int(note_str[2:]) if len(note_str) > 2 else 0
        else:
            note = note_str[0]
            octave = int(note_str[1:]) if len(note_str) > 1 else 0
        
        try:
            note_index = NOTE_NAMES.index(note.upper())
        except ValueError:
            raise ValueError(f"Invalid note: {note_str}")
        
        return 12 + octave * 12 + note_index
    
    start_midi = note_to_midi(start)
    end_midi = note_to_midi(end)
    actual_end = min(end_midi, 127)

    full_range = list(range(start_midi, actual_end + 1))
    
    if len(full_range) <= 16:
        selected = full_range
    else:
        selected = []
        step = (len(full_range) - 1) / 15
        for i in range(16):
            idx = int(i * step)
            selected.append(full_range[idx])
    
    return [(n, get_note_name(n)) for n in selected]

class FileWriterPool:
    """Thread pool for parallel file writing"""
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.lock = threading.Lock()
    
    def write_async(self, path, sample_rate, audio_data):
        """Queue a file write operation"""
        future = self.executor.submit(self._write_file, path, sample_rate, audio_data)
        with self.lock:
            self.futures.append(future)
    
    def _write_file(self, path, sample_rate, audio_data):
        """Actually write the file"""
        try:
            wavfile.write(path, sample_rate, audio_data)
            return True
        except Exception as e:
            print(f"\n✗ Error writing {path}: {e}")
            return False
    
    def wait_all(self):
        """Wait for all pending writes to complete"""
        with self.lock:
            futures_copy = self.futures.copy()
            self.futures.clear()
        
        for future in futures_copy:
            future.result()
    
    def shutdown(self):
        """Shutdown the thread pool"""
        self.wait_all()
        self.executor.shutdown(wait=True)

def detect_silence_end(audio, sample_rate, threshold_db=SILENCE_THRESHOLD_DB, 
                       window_sec=SILENCE_CHECK_WINDOW):
    """Detect where audio effectively ends (becomes silent)"""
    if len(audio) == 0:
        return 0

    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    
    threshold_linear = db_to_linear(threshold_db)
    window_samples = int(sample_rate * window_sec)
    
    for i in range(len(audio_mono) - window_samples, 0, -window_samples):
        window = audio_mono[i:i + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        
        if rms > threshold_linear:
            return min(i + window_samples, len(audio_mono))
    
    return len(audio_mono)

def render_note(engine, synth, midi_note, note_name, use_smart_tail=True):
    """Render a single note"""
    synth.clear_midi()
    synth.add_midi_note(midi_note, NOTE_VELOCITY, 0.0, NOTE_DURATION)
    
    engine.load_graph([(synth, [])])
    
    total_duration = NOTE_DURATION + RENDER_TAIL
    engine.set_bpm(120.0)
    engine.render(total_duration)
    audio = engine.get_audio()
    
    if use_smart_tail:
        silence_start = detect_silence_end(audio.T, SAMPLE_RATE)
        silence_time = silence_start / SAMPLE_RATE
        
        if silence_time > total_duration - 0.1:
            extra_time = min(MAX_TAIL_EXTENSION, 2.0)
            total_duration += extra_time
            
            synth.clear_midi()
            synth.add_midi_note(midi_note, NOTE_VELOCITY, 0.0, NOTE_DURATION)
            engine.load_graph([(synth, [])])
            engine.render(total_duration)
            audio = engine.get_audio()
            silence_start = detect_silence_end(audio.T, SAMPLE_RATE)
    
    audio = audio.T
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16

def render_preset_to_wavs(serum_path, preset_path, output_dir, preset_name, 
                          start_note="C0", end_note="G9", use_smart_tail=True):
    """Render a preset to individual WAV files (only 16 optimal samples)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    notes = get_optimal_16_notes(start_note, end_note)
    
    print(f"Rendering {len(notes)} optimally distributed samples from {start_note} to {end_note}")
    
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor("serum", serum_path)
    
    if preset_path:
        synth.load_preset(preset_path)
    
    writer_pool = FileWriterPool(max_workers=4)
    
    rendered_files = []
    for i, (midi_note, note_name) in enumerate(notes, 1):
        print(progress_bar(i, len(notes), prefix=f"Rendering {preset_name}", 
                          extra=note_name), end='', flush=True)
        
        audio = render_note(engine, synth, midi_note, note_name, use_smart_tail)
        
        filename = f"{preset_name}_{note_name}_{midi_note:03d}.wav"
        output_path = os.path.join(output_dir, filename)
        
        writer_pool.write_async(output_path, SAMPLE_RATE, audio)
        rendered_files.append(output_path)
    
    print()
    
    writer_pool.shutdown()
    
    return rendered_files

class Sample:
    """Represents a sample for XI instrument"""
    def __init__(self, note: str, octave: int, midi: int, path: str):
        self.note = note
        self.octave = octave
        self.midi = midi
        self.path = path
        self.data = b''
        self.length_bytes = 0
        self.loop_start_bytes = 0
        self.loop_length_bytes = 0
        self.relative_note = 0
        self.finetune = 0
        self.volume = 64
        self.panning = 128
        self.sample_type = 0x10
    
    def load(self, max_length=0):
        """Load WAV file and convert to delta-encoded mono 16-bit"""
        with open(self.path, 'rb') as fp:
            chunks = self._read_chunks(fp)
            if b'fmt ' not in chunks or b'data' not in chunks:
                raise ValueError("Missing fmt or data chunk")
            
            fmt = self._parse_fmt(fp, *chunks[b'fmt '])
            data_off, data_size = chunks[b'data']
            
            fp.seek(data_off)
            raw = fp.read(data_size)
            
            if self._is_ieee_float(fmt):
                audio = self._decode_float(raw, fmt)
            else:
                audio = self._decode_pcm(raw, fmt)
            
            if fmt['num_channels'] == 2:
                audio = np.mean(audio.reshape(-1, 2), axis=1)
            elif fmt['num_channels'] > 2:
                audio = np.mean(audio.reshape(-1, fmt['num_channels']), axis=1)
            
            if fmt['sample_rate'] != 44100:
                audio = self._resample(audio, fmt['sample_rate'], 44100)
            
            if max_length > 0 and len(audio) > max_length:
                audio = audio[:max_length]
            
            audio = np.clip(audio, -1.0, 1.0)
            samples_int16 = (audio * 32767).astype(np.int16)
            
            if len(samples_int16) & 1:
                samples_int16 = samples_int16[:-1]
            
            delta = []
            old = 0
            for x in samples_int16:
                d = int(x) - old
                while d > 32767:
                    d -= 65536
                while d < -32768:
                    d += 65536
                delta.append(d)
                old = int(x)
            
            self.data = struct.pack('<' + 'h' * len(delta), *delta)
            self.length_bytes = len(delta) * 2
    
    def _read_chunks(self, fp):
        """Read WAV chunks"""
        fp.seek(0)
        if fp.read(4) != b'RIFF':
            raise ValueError("Not a RIFF file")
        _ = fp.read(4)
        if fp.read(4) != b'WAVE':
            raise ValueError("Not a WAVE file")
        
        chunks = {}
        while True:
            hdr = fp.read(8)
            if len(hdr) < 8:
                break
            cid, size = struct.unpack('<4sI', hdr)
            start = fp.tell()
            chunks[cid] = (start, size)
            fp.seek(start + size + (size & 1))
        return chunks
    
    def _parse_fmt(self, fp, off, size):
        """Parse fmt chunk"""
        fp.seek(off)
        data = fp.read(size)
        if size < 16:
            raise ValueError("WAV fmt chunk too small")
        
        (audio_format, num_channels, sample_rate, byte_rate,
         block_align, bits_per_sample) = struct.unpack_from('<HHIIHH', data, 0)
        
        ext = {}
        if size >= 18:
            cb = struct.unpack_from('<H', data, 16)[0]
            if cb and size >= 18 + cb:
                ext_data = data[18:18+cb]
                if audio_format == 0xFFFE and len(ext_data) >= 24:
                    valid_bits, channel_mask = struct.unpack_from('<HI', ext_data, 0)
                    subformat = ext_data[8:24]
                    ext = {'valid_bits': valid_bits, 'channel_mask': channel_mask, 
                           'subformat': subformat}
        
        return {
            'audio_format': audio_format,
            'num_channels': num_channels,
            'sample_rate': sample_rate,
            'byte_rate': byte_rate,
            'block_align': block_align,
            'bits_per_sample': bits_per_sample,
            'ext': ext
        }
    
    def _is_ieee_float(self, fmt):
        """Check if format is IEEE float"""
        af = fmt['audio_format']
        if af == 3:
            return True
        if af == 0xFFFE and fmt['ext']:
            sub = fmt['ext']['subformat']
            KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = (
                b'\x03\x00\x00\x00\x00\x00\x10\x00'
                b'\x80\x00\x00\xaa\x00\x38\x9b\x71'
            )
            return sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        return False
    
    def _decode_pcm(self, raw, fmt):
        """Decode PCM audio"""
        bits = fmt['bits_per_sample']
        nc = fmt['num_channels']
        
        if bits == 8:
            arr = np.frombuffer(raw, dtype=np.uint8)
            return (arr.astype(np.float32) - 128) / 128.0
        elif bits == 16:
            arr = np.frombuffer(raw, dtype=np.int16)
            return arr.astype(np.float32) / 32768.0
        elif bits == 24:
            frames = len(raw) // (3 * nc)
            arr = np.zeros(frames * nc, dtype=np.int32)
            for i in range(frames * nc):
                b = raw[i*3:(i+1)*3]
                val = int.from_bytes(b, 'little', signed=False)
                if val & 0x800000:
                    val -= 0x1000000
                arr[i] = val
            return arr.astype(np.float32) / 8388608.0
        elif bits == 32:
            arr = np.frombuffer(raw, dtype=np.int32)
            return arr.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bits}")
    
    def _decode_float(self, raw, fmt):
        """Decode IEEE float audio"""
        bits = fmt['bits_per_sample']
        if bits == 32:
            return np.frombuffer(raw, dtype=np.float32)
        elif bits == 64:
            return np.frombuffer(raw, dtype=np.float64).astype(np.float32)
        else:
            raise ValueError(f"Unsupported float bit depth: {bits}")
    
    def _resample(self, audio, from_rate, to_rate):
        """Simple resampling"""
        if from_rate == to_rate:
            return audio
        
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        indices = np.arange(new_length) / ratio
        return np.interp(indices, np.arange(len(audio)), audio)

def parse_filename(fname: str) -> Tuple[str, int, int]:
    """Parse note information from filename"""
    m = NOTE_RE.match(Path(fname).stem)
    if not m:
        raise ValueError(f"Cannot parse note from filename: {fname}")
    note, octave, midi = m.groups()
    return note.upper(), int(octave), int(midi)

def select_samples(samples: List[Sample], max_samples=16) -> List[Sample]:
    """Select up to max_samples evenly distributed"""
    samples.sort(key=lambda s: s.midi)
    
    if len(samples) <= max_samples:
        return samples
    
    step = len(samples) / max_samples
    selected = []
    for i in range(max_samples):
        idx = int(i * step)
        selected.append(samples[idx])
    
    return selected

def compute_relative_note_finetune(samples: List[Sample]):
    """Compute relative note and finetune for each sample"""
    for i, s in enumerate(samples):
        if i == 0:
            s.relative_note = 83 - s.midi
            s.finetune = 0
        elif i == len(samples) - 1 and s.midi > 60:
            s.relative_note = 28
            s.finetune = 104
        else:
            s.relative_note = 88 - s.midi
            s.finetune = 104

def write_xi(samples: List[Sample], inst_name: str, output_path: str):
    """Write XI instrument file with proper format"""
    if not samples:
        raise ValueError("No samples to write")
    
    samples = sorted(samples, key=lambda s: s.midi)
    
    compute_relative_note_finetune(samples)
    
    note_map = [0] * 96
    for note_idx in range(96):
        xi_midi = 12 + note_idx
        best_sample = 0
        for i, s in enumerate(samples):
            if s.midi <= xi_midi:
                best_sample = i
            else:
                break
        note_map[note_idx] = best_sample

    out = bytearray()
    
    out += b'Extended Instrument: '
    inst_bytes = inst_name[:22].ljust(22).encode('ascii', 'replace')
    out += inst_bytes
    out += b'\x1A'
    out += b'FastTracker v2.00   '
    out += struct.pack('<H', 0x0102)

    out += bytes(note_map)
    
    # Volume envelope points (12 pairs of x,y as 2-byte words = 48 bytes)
    vol_points = [(0, 64), (64, 64)]  # Simple sustain envelope
    for i in range(12):
        if i < len(vol_points):
            x, y = vol_points[i]
        else:
            x, y = 0, 0
        out += struct.pack('<HH', x, y)
    
    # Panning envelope points (12 pairs = 48 bytes)
    pan_points = [(0, 32), (64, 32)]  # Center panning
    for i in range(12):
        if i < len(pan_points):
            x, y = pan_points[i]
        else:
            x, y = 0, 0
        out += struct.pack('<HH', x, y)
    
    # Vibrato settings (6 bytes: type, sweep, depth, rate, fadeout)
    out += struct.pack('<BBBBH', 0, 0, 0, 0, 0)
    
    # Envelope counts and flags (10 bytes)
    out += struct.pack('<BBBBBBBBBB',
                       len(vol_points),  # Number of volume points
                       len(pan_points),  # Number of panning points
                       0,  # Volume sustain point
                       0,  # Volume loop start
                       0,  # Volume loop end
                       0,  # Panning sustain point
                       0,  # Panning loop start
                       0,  # Panning loop end
                       0,  # Volume type flags
                       0)  # Panning type flags
    
    # Reserved (22 bytes)
    out += bytes(22)
    
    # Number of samples (2 bytes at offset 0x128)
    out += struct.pack('<H', len(samples))
    
    # Sample headers (40 bytes each)
    for s in samples:
        # Length in BYTES
        out += struct.pack('<I', s.length_bytes)
        # Loop start in BYTES
        out += struct.pack('<I', s.loop_start_bytes)
        # Loop length in BYTES
        out += struct.pack('<I', s.loop_length_bytes)
        # Volume (0-64)
        out += struct.pack('B', s.volume)
        # Finetune (-128 to 127)
        out += struct.pack('b', s.finetune)
        # Type (bit 0: loop, bit 1: pingpong, bit 4: 16-bit)
        out += struct.pack('B', s.sample_type)
        # Panning (0-255)
        out += struct.pack('B', s.panning)
        # Relative note (-128 to 127)
        out += struct.pack('b', s.relative_note)
        # Reserved
        out += bytes([0])
        # Sample name (22 bytes)
        sname = Path(s.path).stem[:22].encode('ascii', 'replace')
        out += sname.ljust(22, b' ')
    
    for s in samples:
        out += s.data
    
    with open(output_path, 'wb') as f:
        f.write(out)

def convert_wavs_to_xi(wav_dir: Path, output_path: Path, inst_name: str, 
                       max_samples=16, show_progress=False):
    """Convert a directory of WAVs to XI instrument"""
    files = {}
    for pat in ("*.wav", "*.WAV"):
        for p in wav_dir.glob(pat):
            files[p.name.lower()] = p
    
    wavs = list(files.values())
    
    if not wavs:
        return False, 0, 0, "No WAV files found"
    
    samples = []
    skipped = 0
    
    for i, wav_path in enumerate(wavs, 1):
        if show_progress:
            print(f"\rLoading samples... {i}/{len(wavs)}", end='', flush=True)
        
        try:
            note, octave, midi = parse_filename(wav_path.name)
            s = Sample(note, octave, midi, str(wav_path))
            s.load()
            samples.append(s)
        except Exception as e:
            skipped += 1
    
    if show_progress:
        print()
    
    if not samples:
        return False, 0, skipped, "No valid samples could be loaded"
    
    samples = select_samples(samples, max_samples)

    try:
        write_xi(samples, inst_name, str(output_path))
        return True, len(samples), skipped, None
    except Exception as e:
        return False, 0, skipped, str(e)

def process_single_fxp(serum_path, fxp_path, output_xi_path, 
                      start_note="C0", end_note="G9", use_smart_tail=True):
    """Process a single FXP file to XI"""
    
    fxp_name = Path(fxp_path).stem
    inst_name = fxp_name[:22]
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nRendering preset: {fxp_name}")
        
        render_preset_to_wavs(
            serum_path=serum_path,
            preset_path=fxp_path,
            output_dir=temp_dir,
            preset_name=fxp_name,
            start_note=start_note,
            end_note=end_note,
            use_smart_tail=use_smart_tail
        )
        
        print(f"Converting to XI instrument...")
        
        success, num_samples, num_skipped, error = convert_wavs_to_xi(
            wav_dir=Path(temp_dir),
            output_path=output_xi_path,
            inst_name=inst_name,
            max_samples=16,
            show_progress=True
        )
        
        if success:
            print(f"✓ Created: {output_xi_path}")
            print(f"  Samples: {num_samples}")
            file_size = output_xi_path.stat().st_size
            print(f"  Size: {file_size / 1024:.1f} KB")
            return True
        else:
            print(f"✗ Failed: {error}")
            return False

def process_bulk_fxp(serum_path, input_dir, output_base_dir, 
                    start_note="C0", end_note="G9", use_smart_tail=True, 
                    mirror_structure=True):
    """Process all FXP files in a directory tree, mirroring structure"""
    
    fxp_files = list(Path(input_dir).rglob("*.fxp"))
    
    if not fxp_files:
        print("✗ No FXP files found")
        return
    
    print(f"\nFound {len(fxp_files)} FXP files")
    print(f"Output base: {output_base_dir}\n")
    
    results = []
    import tempfile
    
    for i, fxp_path in enumerate(fxp_files, 1):
        fxp_name = fxp_path.stem
        
        if mirror_structure:
            rel_path = fxp_path.relative_to(input_dir).parent
            output_dir = Path(output_base_dir) / rel_path
        else:
            output_dir = Path(output_base_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_xi_path = output_dir / f"{fxp_name}.xi"
        
        print(f"\n[{i}/{len(fxp_files)}] {fxp_path.relative_to(input_dir)}")
        print(f"    → {output_xi_path.relative_to(output_base_dir)}")
        
        if output_xi_path.exists():
            print("    ⏭ Skipping (already exists)")
            results.append((str(fxp_path), False, "Already exists"))
            continue
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                render_preset_to_wavs(
                    serum_path=serum_path,
                    preset_path=str(fxp_path),
                    output_dir=temp_dir,
                    preset_name=fxp_name,
                    start_note=start_note,
                    end_note=end_note,
                    use_smart_tail=use_smart_tail
                )
                
                success, num_samples, num_skipped, error = convert_wavs_to_xi(
                    wav_dir=Path(temp_dir),
                    output_path=output_xi_path,
                    inst_name=fxp_name[:22],
                    max_samples=16,
                    show_progress=False
                )
                
                if success:
                    file_size = output_xi_path.stat().st_size
                    print(f"    ✓ Success ({num_samples} samples, {file_size / 1024:.1f} KB)")
                    results.append((str(fxp_path), True, None))
                else:
                    print(f"    ✗ Failed: {error}")
                    results.append((str(fxp_path), False, error))
            
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append((str(fxp_path), False, str(e)))
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}\n")
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for path, success, error in results:
            if not success:
                print(f"  ✗ {Path(path).name}: {error}")
    
    print(f"\nOutput location: {output_base_dir}\n")

def main():
    """Main interactive interface"""
    show_header()
    
    show_section(1, 4, "Serum VST Detection", clear=False)
    
    print("Searching for Serum installations...\n")
    versions = find_all_serum_versions()
    
    if not versions:
        print("✗ No Serum installations found")
        print("\nPlease install Serum and ensure it's in a standard VST location.")
        input("\nPress Enter to exit...")
        return
    
    print(f"Found {len(versions)} Serum installation(s):\n")
    
    for i, v in enumerate(versions, 1):
        status = "✓ Accessible" if v['accessible'] else "✗ Locked"
        print(f"  {i}. {Path(v['path']).name}")
        print(f"     {v['path']}")
        print(f"     {v['type']} | {v['arch']} | {status}\n")
    
    if len(versions) == 1:
        serum_path = versions[0]['path']
        print(f"Using: {Path(serum_path).name}")
    else:
        while True:
            choice = input(f"Select Serum version (1-{len(versions)}) ❯ ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(versions):
                    serum_path = versions[idx]['path']
                    break
            except:
                pass
            print("Invalid choice, try again.")

    print(f"\nTesting Serum load...", end=" ", flush=True)
    if test_vst_load(serum_path):
        print("✓")
    else:
        print("✗ Failed to load")
        input("\nPress Enter to exit...")
        return
    
    input("\nPress Enter to continue...")

    show_section(2, 4, "Processing Mode")
    
    print("Choose processing mode:\n")
    print("  1. Single FXP file")
    print("  2. Bulk process (folder with FXP files)")
    print()
    
    mode = input("Choose (1-2) ❯ ").strip()
    
    if mode == "1":
        print("\nEnter path to .fxp preset file:")
        fxp_path = input("❯ ").strip().strip('"')
        
        if not Path(fxp_path).exists():
            print("✗ File not found")
            input("\nPress Enter to exit...")
            return
        
        print("\nEnter output .xi file path:")
        output_path = input("❯ ").strip().strip('"')
        output_path = Path(output_path)
        
        if output_path.suffix.lower() != '.xi':
            output_path = output_path.with_suffix('.xi')
        
        show_section(3, 4, "Note Range")
        
        print("Select note range (FastTracker uses max 16 samples):\n")
        print("  1. Full (C0-G9) - 16 samples evenly distributed")
        print("  2. Piano (A0-C8) - 16 samples evenly distributed")
        print("  3. Bass (C0-C4) - up to 16 samples")
        print("  4. Lead (C3-C7) - up to 16 samples")
        print()
        
        range_choice = input("Choose (1-4) ❯ ").strip()
        ranges = {
            "1": ("C0", "G9"),
            "2": ("A0", "C8"),
            "3": ("C0", "C4"),
            "4": ("C3", "C7")
        }
        start_note, end_note = ranges.get(range_choice, ("C0", "G9"))
        
        optimal_notes = get_optimal_16_notes(start_note, end_note)
        print(f"\n✓ Will render {len(optimal_notes)} optimally distributed samples")
        print(f"  Range: {start_note} to {end_note}")
        if len(optimal_notes) < 16:
            print(f"  (Range contains {len(optimal_notes)} notes total)")
        
        input("\nPress Enter to continue...")
        
        show_section(4, 4, "Processing")
        
        success = process_single_fxp(
            serum_path=serum_path,
            fxp_path=fxp_path,
            output_xi_path=output_path,
            start_note=start_note,
            end_note=end_note,
            use_smart_tail=True
        )
        
        if success:
            print("\nDone! Your FastTracker instrument is ready!")
        
    elif mode == "2":
        print("\nEnter folder containing .fxp files (will search subdirectories):")
        input_dir = input("❯ ").strip().strip('"')
        
        if not Path(input_dir).exists():
            print("✗ Folder not found")
            input("\nPress Enter to exit...")
            return
        
        print("\nEnter output base directory:")
        output_dir = input("❯ ").strip().strip('"')
        
        show_section(3, 4, "Note Range")
        
        print("Select note range (FastTracker uses max 16 samples):\n")
        print("  1. Full (C0-G9) - 16 samples evenly distributed")
        print("  2. Piano (A0-C8) - 16 samples evenly distributed")
        print("  3. Bass (C0-C4) - up to 16 samples")
        print("  4. Lead (C3-C7) - up to 16 samples")
        print()
        
        range_choice = input("Choose (1-4) ❯ ").strip()
        ranges = {
            "1": ("C0", "G9"),
            "2": ("A0", "C8"),
            "3": ("C0", "C4"),
            "4": ("C3", "C7")
        }
        start_note, end_note = ranges.get(range_choice, ("C0", "G9"))
        
        optimal_notes = get_optimal_16_notes(start_note, end_note)
        print(f"\n✓ Will render {len(optimal_notes)} optimally distributed samples per preset")
        print(f"  Range: {start_note} to {end_note}")
        if len(optimal_notes) < 16:
            print(f"  (Range contains {len(optimal_notes)} notes total)")
        
        input("\nPress Enter to continue...")
        
        show_section(4, 4, "Bulk Processing")
        
        process_bulk_fxp(
            serum_path=serum_path,
            input_dir=input_dir,
            output_base_dir=output_dir,
            start_note=start_note,
            end_note=end_note,
            use_smart_tail=True,
            mirror_structure=True
        )
        
        print("\nDone! Your FastTracker instruments are ready!")
    
    else:
        print("✗ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")
