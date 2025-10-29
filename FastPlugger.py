#!/usr/bin/env python3
"""
FastPlugger - Convert WAV keymaps to Fast Tracker 2 XI files
"""

import os, re, struct, sys, time
from pathlib import Path
from typing import List, Tuple, Optional

NOTE_RE = re.compile(r'.*_([A-G]#?)(\d+)_(\d+)$')

ASCII_HEADER = r"""
 _______              ______  _                               
(_______)        _   (_____ \| |                              
 _____ ____  ___| |_  _____) ) |_   _  ____  ____  ____  ____ 
|  ___) _  |/___)  _)|  ____/| | | | |/ _  |/ _  |/ _  )/ ___)
| |  ( ( | |___ | |__| |     | | |_| ( ( | ( ( | ( (/ /| |    
|_|   \_||_(___/ \___)_|     |_|\____|\_|| |\_|| |\____)_|    
                                     (_____(_____|            
"""

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_header():
    """Show the ASCII header"""
    print(ASCII_HEADER)

def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    """Prompt for input with optional default"""
    if default:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "
    
    while True:
        response = input(prompt_text).strip()
        if response:
            return response
        elif default:
            return default
        else:
            print("This field is required. Please enter a value.\n")

def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no with default"""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'\n")

def prompt_choice(prompt: str, choices: List[str], default: int = 0) -> int:
    """Prompt for a choice from a list"""
    print(prompt)
    for i, choice in enumerate(choices, 1):
        default_marker = " (default)" if i == default + 1 else ""
        print(f"  {i}. {choice}{default_marker}")
    print()
    
    while True:
        response = input(f"Enter choice [1-{len(choices)}]: ").strip()
        if not response:
            return default
        try:
            choice = int(response)
            if 1 <= choice <= len(choices):
                return choice - 1
            print(f"Please enter a number between 1 and {len(choices)}\n")
        except ValueError:
            print(f"Please enter a number between 1 and {len(choices)}\n")

def validate_directory(path_str: str) -> Optional[Path]:
    """Validate that a directory exists"""
    path = Path(path_str.strip('"').strip("'"))
    if not path.exists():
        print(f"\nError: Directory not found: {path}")
        print("Please try again.\n")
        return None
    if not path.is_dir():
        print(f"\nError: Not a directory: {path}")
        print("Please try again.\n")
        return None
    return path

def find_all_wav_folders(base_path: Path) -> Tuple[bool, List[Path]]:
    """
    Recursively find ALL directories containing WAV files at any depth.
    Returns (has_wavs_in_base, list_of_all_folders_with_wavs)
    """
    direct_wavs = list(base_path.glob('*.wav')) + list(base_path.glob('*.WAV'))
    has_direct_wavs = len(direct_wavs) > 0
    
    folders_with_wavs = []
    
    def scan_directory(path: Path):
        """Recursively scan for folders with WAV files"""
        try:
            for item in path.iterdir():
                if item.is_dir():
                    wav_files = list(item.glob('*.wav')) + list(item.glob('*.WAV'))
                    if wav_files:
                        folders_with_wavs.append(item)
                    scan_directory(item)
        except PermissionError:
            pass
    
    scan_directory(base_path)
    
    return has_direct_wavs, folders_with_wavs

def validate_output_path(path_str: str) -> Optional[Path]:
    """Validate output file path"""
    path = Path(path_str.strip('"').strip("'"))
    
    if path.suffix.lower() != '.xi':
        path = path.with_suffix('.xi')
        print(f"\nAdding .xi extension: {path.name}")
    
    parent = path.parent
    if not parent.exists():
        print(f"\nError: Directory does not exist: {parent}")
        print("Please try again.\n")
        return None
    
    if path.exists():
        if not prompt_yes_no(f"\nFile already exists. Overwrite?", default=False):
            print("Please choose a different filename.\n")
            return None
    
    return path

def show_loading_bar(current: int, total: int, prefix: str = "", length: int = 40):
    """Display a loading bar"""
    percent = 100 * (current / float(total))
    filled = int(length * current // total)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r{prefix} [{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()

def _read_chunks(fp):
    fp.seek(0)
    if fp.read(4) != b'RIFF': raise ValueError("Not a RIFF file")
    _ = fp.read(4)
    if fp.read(4) != b'WAVE': raise ValueError("Not a WAVE file")
    chunks = {}
    while True:
        hdr = fp.read(8)
        if len(hdr) < 8: break
        cid, size = struct.unpack('<4sI', hdr)
        start = fp.tell()
        chunks[cid] = (start, size)
        fp.seek(start + size + (size & 1))
    return chunks

def _parse_fmt(fp, off, size):
    fp.seek(off); data = fp.read(size)
    if size < 16: raise ValueError("WAV fmt chunk too small")
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
                ext = {'valid_bits': valid_bits, 'channel_mask': channel_mask, 'subformat': subformat}
    return {'audio_format': audio_format, 'num_channels': num_channels,
            'sample_rate': sample_rate, 'byte_rate': byte_rate,
            'block_align': block_align, 'bits_per_sample': bits_per_sample, 'ext': ext}

def _is_ieee_float(fmt):
    af = fmt['audio_format']
    if af == 0x0003: return True
    if af == 0xFFFE:
        sub = fmt['ext'].get('subformat', b'')
        return sub.startswith(b'\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71')
    return False

def _is_pcm(fmt):
    af = fmt['audio_format']
    if af == 0x0001: return True
    if af == 0xFFFE:
        sub = fmt['ext'].get('subformat', b'')
        return sub.startswith(b'\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71')
    return False

def _to_mono_int16(raw, fmt):
    import array, struct as st
    ch = fmt['num_channels']; bps = fmt['bits_per_sample']

    if _is_ieee_float(fmt):
        if bps == 32:
            vals = [v[0] for v in st.iter_unpack('<f', raw)]
        elif bps == 64:
            vals = [v[0] for v in st.iter_unpack('<d', raw)]
        else:
            raise ValueError(f"Unsupported float bps={bps}")
        if ch > 1:
            vals = [sum(vals[i:i+ch])/ch for i in range(0, len(vals), ch)]
        out = array.array('h')
        for v in vals:
            if v < -1.0: v = -1.0
            if v >  1.0: v =  1.0
            out.append(int(v * 32767.0))
        return out

    if not _is_pcm(fmt):
        raise ValueError("Unsupported WAV format (not PCM/float)")

    if bps == 8:
        arr = array.array('B', raw)
        mono = [(sum(arr[i:i+ch])//ch) for i in range(0, len(arr), ch)] if ch > 1 else arr
        return array.array('h', ((x - 128) << 8 for x in mono))

    if bps == 16:
        arr = array.array('h'); arr.frombytes(raw)
        if ch > 1:
            mono = []
            for i in range(0, len(arr), ch):
                s = 0
                for k in range(ch): s += arr[i+k]
                mono.append(int(s / ch))
        else:
            mono = arr
        return array.array('h', mono)

    if bps == 24:
        out = array.array('h'); step = ch * 3
        for i in range(0, len(raw), step):
            s = 0
            for k in range(ch):
                b0,b1,b2 = raw[i+3*k:i+3*k+3]
                v = b0 | (b1<<8) | (b2<<16)
                if v & 0x800000: v -= 0x1000000
                s += v
            out.append(int((s / ch) >> 8))
        return out

    if bps == 32:
        arr = array.array('i'); arr.frombytes(raw)
        if ch > 1:
            mono = []
            for i in range(0, len(arr), ch):
                s = 0
                for k in range(ch): s += arr[i+k]
                mono.append(int(s / ch))
        else:
            mono = arr
        return array.array('h', (v >> 16 for v in mono))

    raise ValueError(f"Unsupported PCM bits_per_sample={bps}")

def load_wav_any(path: str):
    with open(path, 'rb') as fp:
        chunks = _read_chunks(fp)
        if b'fmt ' not in chunks or b'data' not in chunks:
            raise ValueError("WAV missing fmt or data chunk")
        f_off, f_size = chunks[b'fmt ']; d_off, d_size = chunks[b'data']
        fmt = _parse_fmt(fp, f_off, f_size)
        fp.seek(d_off); raw = fp.read(d_size)
    return _to_mono_int16(raw, fmt)

class Sample:
    def __init__(self, note: str, octave: int, midi: int, path: str):
        self.note, self.octave, self.midi, self.path = note, octave, midi, path
        self.sample_type = 0x10
        self.volume, self.panning = 64, 128
        self.finetune, self.relative_note = 0, 0

        self.data = b''
        self.length_bytes = 0
        self.loop_start_bytes = 0
        self.loop_length_bytes = 0

    def load(self, max_frames: int = 0):
        smp = load_wav_any(self.path)
        if max_frames and len(smp) > max_frames:
            smp = smp[:max_frames]
        if len(smp) & 1:
            smp = smp[:-1]

        delta = []
        old = 0
        for x in smp:
            if x < -32768: x = -32768
            if x >  32767: x =  32767
            d = int(x) - old
            while d >  32767: d -= 65536
            while d < -32768: d += 65536
            delta.append(d); old = int(x)

        self.data = struct.pack('<' + 'h'*len(delta), *delta)
        self.length_bytes = len(delta) * 2
        self.sample_type = 0x10

    def header_bytes(self) -> bytes:
        nm = os.path.basename(self.path)[:22].ljust(22, '\x00').encode('ascii', 'replace')
        b = bytearray()
        b += struct.pack('<I', self.length_bytes)
        b += struct.pack('<I', self.loop_start_bytes)
        b += struct.pack('<I', self.loop_length_bytes)
        b += struct.pack('B', self.volume)
        b += struct.pack('b', self.finetune)
        b += struct.pack('B', self.sample_type)
        b += struct.pack('B', self.panning)
        b += struct.pack('b', self.relative_note)
        b += b'\x00'
        b += nm
        return bytes(b)

def parse_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse filename to extract note, octave, and MIDI number.
    
    Format: ANYTHING_NOTE+OCTAVE_MIDI.wav
    Where NOTE is A-G (with optional #), OCTAVE is 0-9, MIDI is 012-127
    """
    name = os.path.splitext(filename)[0]
    m = NOTE_RE.match(name)
    if not m:
        raise ValueError(
            f"Filename doesn't match required pattern.\n"
            f"Expected: ANYTHING_NOTE+OCTAVE_MIDI.wav\n"
            f"Examples:\n"
            f"  - INIT_C4_060.wav\n"
            f"  - My Preset_A#0_022.wav\n"
            f"  - CHORD - Such 90s_C4_060.wav\n"
            f"Got: {filename}"
        )
    return m.group(1), int(m.group(2)), int(m.group(3))

def select_samples(samples: List[Sample], max_samples: int) -> List[Sample]:
    valid = [s for s in samples if 12 <= s.midi <= 107] or samples
    valid.sort(key=lambda s: s.midi)
    if len(valid) <= max_samples: return valid
    mn, mx = valid[0].midi, valid[-1].midi
    step = (mx - mn) / max(1, (max_samples - 1))
    picked = []
    for i in range(max_samples):
        target = mn + i*step
        best = min(valid, key=lambda s: abs(s.midi - target))
        if best not in picked: picked.append(best)
    for s in valid:
        if len(picked) >= max_samples: break
        if s not in picked: picked.append(s)
    picked.sort(key=lambda s: s.midi)
    return picked[:max_samples]

def build_note_map(samples: List[Sample]) -> List[int]:
    note_map = [0]*96
    for xi_note in range(96):
        xi_midi = 12 + xi_note
        best = 0
        for idx, s in enumerate(samples):
            if s.midi <= xi_midi: best = idx
            else: break
        note_map[xi_note] = best
    return note_map

def apply_rel_and_ft(samples: List[Sample]):
    n = len(samples)
    for i, s in enumerate(samples):
        if i == 0:
            s.relative_note = max(-127, min(127, 83 - s.midi))
            s.finetune = 0
        elif i == n-1 and s.midi > 60:
            s.relative_note = 28
            s.finetune = 104
        else:
            s.relative_note = max(-127, min(127, 88 - s.midi))
            s.finetune = 104

def write_xi(samples: List[Sample], inst_name: str, out_path: str):
    samples = sorted(samples, key=lambda s: s.midi)
    apply_rel_and_ft(samples)
    note_map = build_note_map(samples)

    out = bytearray()
    out += b'Extended Instrument: '
    out += inst_name[:22].ljust(22).encode('ascii', 'replace')
    out += b'\x1A'
    out += b'wav2xi len-bytes fix'[:20].ljust(20, b' ')
    out += struct.pack('<H', 0x0102)

    out += bytes(note_map)

    vol_points = [(0,64),(64,64)]
    for i in range(12):
        x,y = vol_points[i] if i < len(vol_points) else (0,0)
        out += struct.pack('<HH', x, y)

    pan_points = [(0,32),(64,32)]
    for i in range(12):
        x,y = pan_points[i] if i < len(pan_points) else (0,0)
        out += struct.pack('<HH', x, y)

    out += struct.pack('<BBBBH', 0,0,0,0, 0)
    out += struct.pack('<BBBBBBBBBB',
                       len(vol_points), len(pan_points),
                       0, 0, 0, 0, 0, 0, 0, 0)
    out += b'\x00' * 22

    out += struct.pack('<H', len(samples))

    for s in samples:
        out += s.header_bytes()

    for s in samples:
        out += s.data

    assert out[0x128] == (len(samples) & 0xFF) and out[0x129] == (len(samples) >> 8)

    with open(out_path, 'wb') as f:
        f.write(out)

def process_folder(input_dir: Path, output_path: Path, inst_name: str, show_progress: bool = True) -> Tuple[bool, int, int, str]:
    """
    Process a single folder of WAV files.
    Returns (success, num_samples, num_skipped, error_message)
    """
    files = {}
    for pat in ("*.wav", "*.WAV"):
        for p in input_dir.glob(pat):
            files[p.name.lower()] = p
    wavs = list(files.values())
    
    if not wavs:
        return False, 0, 0, "No WAV files found"
    
    samples: List[Sample] = []
    skipped = []
    
    for i, p in enumerate(wavs, 1):
        try:
            note, octv, midi = parse_filename(p.name)
            s = Sample(note, octv, midi, str(p))
            s.load(0)
            samples.append(s)
            if show_progress:
                show_loading_bar(i, len(wavs), "Progress")
        except Exception as e:
            skipped.append((p.name, str(e)))
            if show_progress:
                show_loading_bar(i, len(wavs), "Progress")
    
    if show_progress:
        print()
    
    if not samples:
        return False, 0, len(skipped), "No valid samples could be loaded"
    
    max_samples = 16
    samples = select_samples(samples, max_samples)
    
    write_xi(samples, inst_name, str(output_path))
    
    return True, len(samples), len(skipped), ""

def interactive_mode():
    """Run in interactive mode"""
    clear_screen()
    show_header()
    
    print("This tool will convert a folder of WAV files into a FastTracker II")
    print("Extended Instrument (.xi) file.\n")
    print("Supported file naming patterns:")
    print("  - INIT_C4_060.wav")
    print("  - My Preset_A#0_022.wav")
    print("  - CHORD - Such 90s_C4_060.wav")
    print("  - TSP_SDG_chord_legato_G#5_080.wav")
    print("  - And any similar pattern ending with: _NOTE+OCTAVE_MIDI.wav\n")
    print("=" * 70)
    print()
    
    input_dir = None
    while not input_dir:
        path_str = prompt_input("Enter folder path")
        input_dir = validate_directory(path_str)
    
    print("\nScanning for folders with WAV files...")
    has_direct_wavs, all_wav_folders = find_all_wav_folders(input_dir)
    
    process_mode = "single"
    
    if has_direct_wavs and all_wav_folders:
        clear_screen()
        show_header()
        print(f"Selected folder: {input_dir}")
        print()
        print("This folder contains WAV files directly AND subdirectories with WAVs.")
        print()
        choice = prompt_choice(
            "What would you like to process?",
            [
                "Process WAVs in main folder only",
                f"Bulk process all {len(all_wav_folders)} subdirectories with WAVs"
            ],
            default=0
        )
        process_mode = "single" if choice == 0 else "bulk"
        
    elif all_wav_folders and not has_direct_wavs:
        clear_screen()
        show_header()
        print(f"Selected folder: {input_dir}")
        print()
        print(f"Found {len(all_wav_folders)} subdirectories with WAV files:")
        
        for folder in all_wav_folders[:10]:
            rel_path = folder.relative_to(input_dir)
            print(f"  - {rel_path}")
        if len(all_wav_folders) > 10:
            print(f"  ... and {len(all_wav_folders) - 10} more")
        print()
        
        if len(all_wav_folders) == 1:
            process_mode = "single"
            input_dir = all_wav_folders[0]
            print(f"Processing: {input_dir.name}\n")
        else:
            choice = prompt_choice(
                "What would you like to do?",
                [
                    "Process a single subdirectory",
                    f"Bulk process all {len(all_wav_folders)} subdirectories"
                ],
                default=1
            )
            
            if choice == 0:
                process_mode = "single"
                clear_screen()
                show_header()
                print("Select subdirectory to process:\n")
                for i, folder in enumerate(all_wav_folders, 1):
                    rel_path = folder.relative_to(input_dir)
                    print(f"  {i}. {rel_path}")
                print()
                
                folder_names = [str(f.relative_to(input_dir)) for f in all_wav_folders]
                subdir_choice = prompt_choice("", folder_names, default=0)
                input_dir = all_wav_folders[subdir_choice]
            else:
                process_mode = "bulk"
    
    elif has_direct_wavs and not all_wav_folders:
        process_mode = "single"
        wav_count = len(list(input_dir.glob('*.wav'))) + len(list(input_dir.glob('*.WAV')))
        print(f"\nFound {wav_count} WAV file(s) in directory.")
    
    else:
        print("\nError: No WAV files found in this directory or its subdirectories.")
        return 1
    
    if process_mode == "single":
        return process_single_folder(input_dir)
    else:
        return process_bulk_folders(input_dir, all_wav_folders)

def process_single_folder(input_dir: Path) -> int:
    """Process a single folder interactively"""
    clear_screen()
    show_header()
    print(f"Input folder: {input_dir}")
    print()
    
    suggested_name = input_dir.name + ".xi"
    output_path = None
    while not output_path:
        path_str = prompt_input("Enter output filename", default=suggested_name)
        output_path = validate_output_path(path_str)
    
    clear_screen()
    show_header()
    print(f"Input folder: {input_dir}")
    print(f"Output file: {output_path}")
    print()
    
    inst_name = prompt_input("Enter instrument name (max 22 characters)", 
                             default=input_dir.name[:22])[:22]
    
    clear_screen()
    show_header()
    print(f"Input folder: {input_dir}")
    print(f"Output file: {output_path}")
    print(f"Instrument name: {inst_name}")
    print()
    print("=" * 70)
    print()
    
    print("Scanning for WAV files...")
    files = {}
    for pat in ("*.wav", "*.WAV"):
        for p in input_dir.glob(pat):
            files[p.name.lower()] = p
    wavs = list(files.values())
    print(f"Found {len(wavs)} WAV file(s)\n")
    
    print("Loading and processing samples...\n")
    success, num_samples, num_skipped, error = process_folder(input_dir, output_path, inst_name, show_progress=True)
    
    if not success:
        print(f"\nError: {error}")
        return 1
    
    file_size = output_path.stat().st_size
    
    print()
    print("=" * 70)
    print()
    print("SUCCESS! Instrument created successfully.")
    print()
    print(f"Output file: {output_path}")
    print(f"Instrument name: {inst_name}")
    print(f"Samples included: {num_samples}")
    print(f"File size: {file_size / 1024:.1f} KB")
    
    if num_skipped > 0:
        print(f"\nWarning: {num_skipped} file(s) were skipped due to errors.")
    
    print()
    return 0

def process_bulk_folders(base_dir: Path, all_folders: List[Path]) -> int:
    """Process multiple folders in bulk"""
    clear_screen()
    show_header()
    print(f"Base folder: {base_dir}")
    print(f"Subdirectories to process: {len(all_folders)}")
    print()
    
    output_dir = base_dir
    if prompt_yes_no("Save .xi files to the same base folder?", default=True):
        output_dir = base_dir
    else:
        output_dir = None
        while not output_dir:
            path_str = prompt_input("Enter output directory")
            output_dir = validate_directory(path_str)
    
    clear_screen()
    show_header()
    print(f"Bulk Processing: {len(all_folders)} folders")
    print(f"Output directory: {output_dir}")
    print()
    print("=" * 70)
    print()
    
    results = []
    for i, subdir in enumerate(all_folders, 1):
        rel_path = subdir.relative_to(base_dir)
        print(f"\n[{i}/{len(all_folders)}] Processing: {rel_path}")
        print("-" * 70)
        
        output_filename = f"{subdir.name}.xi"
        output_path = output_dir / output_filename
        inst_name = subdir.name[:22]
        
        print(f"Output: {output_filename}")
        print(f"Instrument name: {inst_name}")
        print()

        if output_path.exists():
            print(f"Skipping (file already exists)")
            results.append((str(rel_path), False, 0, 0, "File already exists"))
            continue
        
        success, num_samples, num_skipped, error = process_folder(subdir, output_path, inst_name, show_progress=True)
        
        if success:
            file_size = output_path.stat().st_size
            print(f"Success: {num_samples} samples, {file_size / 1024:.1f} KB")
            results.append((str(rel_path), True, num_samples, num_skipped, ""))
        else:
            print(f"Failed: {error}")
            results.append((str(rel_path), False, 0, num_skipped, error))
    
    print()
    print("=" * 70)
    print()
    print("BULK PROCESSING COMPLETE")
    print()
    
    successful = sum(1 for r in results if r[1])
    failed = len(results) - successful
    
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed folders:")
        for name, success, _, _, error in results:
            if not success:
                print(f"  - {name}: {error}")
        print()
    
    print(f"Output location: {output_dir}")
    print()
    
    return 0

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        import argparse
        ap = argparse.ArgumentParser(description="Convert WAVs to FastTracker 2 .xi instrument")
        ap.add_argument("input_dir", help='Directory with WAVs named like NAME_NOTE_OCT_MIDI.wav')
        ap.add_argument("output_file", help="Output .xi path")
        ap.add_argument("-n","--name", default="Instrument", help="Instrument name (22 chars)")
        ap.add_argument("-m","--max-samples", type=int, default=16, help="Max samples")
        ap.add_argument("-l","--max-length", type=int, default=0, help="Max frames (0=keep full)")
        ap.add_argument("-v","--verbose", action="store_true")
        args = ap.parse_args()
        
        ip = Path(args.input_dir)
        if not ip.exists():
            print("Error: input directory not found")
            return 1
        
        files = {}
        for pat in ("*.wav","*.WAV"):
            for p in ip.glob(pat):
                files[p.name.lower()] = p
        wavs = list(files.values())
        
        if not wavs:
            print("Error: no WAV files found")
            return 1
        
        samples: List[Sample] = []
        for p in wavs:
            try:
                note, octv, midi = parse_filename(p.name)
                s = Sample(note, octv, midi, str(p))
                s.load(args.max_length)
                samples.append(s)
            except Exception as e:
                if args.verbose:
                    print(f"SKIP {p.name}: {e}")
        
        if not samples:
            print("Error: no valid samples parsed")
            return 1
        
        samples = select_samples(samples, args.max_samples)
        write_xi(samples, args.name, args.output_file)
        
        if args.verbose:
            print("\nSample Details:")
            print(f"{'Index':<6} {'File':<24} {'MIDI':<5} {'Rel':<4} {'Fine':<5} {'Size':<10}")
            print("-" * 66)
            for i, s in enumerate(samples):
                fname = Path(s.path).name[:22]
                size_kb = s.length_bytes / 1024
                print(f"{i:<6} {fname:<24} {s.midi:<5} {s.relative_note:<4} {s.finetune:<5} {size_kb:>7.1f} KB")
        
        print(f"\nCreated: {args.output_file}")
        return 0
    
    try:
        return interactive_mode()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.\n")
        return 1
    except Exception as e:
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
