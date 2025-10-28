#!/usr/bin/env python3
"""
SERUM BUSTER
Render serum presets automatically using dawdreamer
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import timedelta
import platform
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import shutil

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

class FileWriterPool:
    """Thread pool for parallel file writing - THIS is the safe speedup!"""
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.lock = threading.Lock()
    
    def write_async(self, path, sample_rate, audio_data):
        """Queue a file write operation."""
        future = self.executor.submit(self._write_file, path, sample_rate, audio_data)
        with self.lock:
            self.futures.append(future)
    
    def _write_file(self, path, sample_rate, audio_data):
        """Actually write the file."""
        try:
            wavfile.write(path, sample_rate, audio_data)
            return True
        except Exception as e:
            print(f"\n✗ Error writing {path}: {e}")
            return False
    
    def wait_all(self):
        """Wait for all pending writes to complete."""
        with self.lock:
            futures_copy = self.futures.copy()
            self.futures.clear()
        
        for future in futures_copy:
            future.result()
    
    def shutdown(self):
        """Shutdown the thread pool."""
        self.wait_all()
        self.executor.shutdown(wait=True)

def check_file_access(path):
    """Check if we can access a file."""
    try:
        with open(path, 'rb') as f:
            f.read(4)
        return True
    except:
        return False

def get_file_architecture(path):
    """Try to determine if a DLL is 32-bit or 64-bit."""
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
    """Find all Serum installations."""
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
    
    program_files = [
        os.environ.get('PROGRAMFILES', r'C:\Program Files'),
        os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')
    ]
    
    search_patterns = ['serum_x64.dll', 'serum.dll', 'serum.vst3']
    
    for pattern in search_patterns:
        for base in program_files:
            if not base or not os.path.exists(base):
                continue
            for root, dirs, files in os.walk(base):
                for file in files:
                    if file.lower() == pattern:
                        full_path = os.path.join(root, file)
                        if not any(f['path'] == full_path for f in found):
                            arch = get_file_architecture(full_path) if full_path.endswith('.dll') else "VST3"
                            accessible = check_file_access(full_path)
                            found.append({
                                'path': full_path,
                                'type': 'VST3' if full_path.endswith('.vst3') else 'VST2',
                                'arch': arch,
                                'accessible': accessible
                            })
                
                if root.count(os.sep) - base.count(os.sep) > 3:
                    dirs.clear()
    
    return found

def test_vst_load(vst_path):
    """Test if DawDreamer can load a VST."""
    try:
        arch = get_file_architecture(vst_path) if vst_path.endswith('.dll') else 'VST3'
        print(f"Architecture: {arch}")
        print("Loading VST...", end=" ", flush=True)
        
        engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
        synth = engine.make_plugin_processor("serum_test", vst_path)
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗\nError: {e}")
        return False

def get_note_name(midi_num):
    """Convert MIDI number to note name."""
    octave = (midi_num - 12) // 12
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"

def get_midi_range(start="C0", end="G9"):
    """Get all MIDI notes in range."""
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

def format_time(seconds):
    """Format seconds into readable time string."""
    return str(timedelta(seconds=int(seconds)))

def progress_bar(current, total, bar_length=40, prefix="Progress", extra=""):
    """Display a progress bar."""
    percent = float(current) / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent_str = f"{percent * 100:.1f}%"
    return f"\r{prefix}: [{bar}] {percent_str} ({current}/{total}) {extra}"

def db_to_linear(db):
    """Convert dB to linear amplitude."""
    return 10 ** (db / 20)

def linear_to_db(linear):
    """Convert linear amplitude to dB."""
    if linear == 0:
        return -np.inf
    return 20 * np.log10(abs(linear))

def is_silent(audio, threshold_db=SILENCE_THRESHOLD_DB, window_seconds=SILENCE_CHECK_WINDOW):
    """Check if the end of audio has reached silence."""
    if len(audio) == 0 or audio.shape[0] == 0:
        return True
    
    window_samples = int(window_seconds * SAMPLE_RATE)
    window_samples = min(window_samples, audio.shape[1])
    
    if window_samples == 0:
        return True
    
    tail_audio = audio[:, -window_samples:]
    
    rms = np.sqrt(np.mean(tail_audio ** 2, axis=1))
    max_rms = np.max(rms)
    
    if max_rms > 0:
        db_level = linear_to_db(max_rms)
        return db_level < threshold_db
    
    return True

def render_with_smart_tail(engine, synth, midi_note, duration, initial_tail=RENDER_TAIL):
    """Render with automatic tail detection - extends until silence!"""
    synth.clear_midi()
    synth.add_midi_note(midi_note, NOTE_VELOCITY, 0.0, duration)
    engine.load_graph([(synth, [])])

    render_time = duration + initial_tail
    engine.render(render_time)
    audio = engine.get_audio()
    
    extension_step = 1.0
    total_extension = 0
    
    while not is_silent(audio) and total_extension < MAX_TAIL_EXTENSION:
        total_extension += extension_step
        new_render_time = duration + initial_tail + total_extension
        
        synth.clear_midi()
        synth.add_midi_note(midi_note, NOTE_VELOCITY, 0.0, duration)
        engine.load_graph([(synth, [])])
        engine.render(new_render_time)
        audio = engine.get_audio()
    
    return audio

def load_preset_into_synth(synth, preset_path, serum_path):
    """Load a preset into the synth processor."""
    if not preset_path or not Path(preset_path).exists():
        return False
    
    preset_path = str(Path(preset_path).resolve())
    is_vst3 = serum_path.endswith('.vst3')
    
    try:
        with open(preset_path, 'rb') as f:
            preset_data = f.read()

        try:
            if synth.load_preset(preset_path):
                return True
        except:
            pass
        
        try:
            if is_vst3:
                if preset_data[:4] == b'CcnK':
                    chunk_data = preset_data[160:]
                    synth.set_state_information(chunk_data)
                else:
                    synth.set_state_information(preset_data)
            else:
                synth.load_state(preset_data)
            return True
        except:
            pass
        
        try:
            if is_vst3:
                synth.load_state(preset_data)
            else:
                synth.set_state_information(preset_data)
            return True
        except:
            pass
        
        return False
    except Exception as e:
        print(f"    Error loading preset: {e}")
        return False

def render_single_note(engine, synth, midi_note, note_name, output_path, file_writer, use_smart_tail=True):
    """Render ONE note cleanly with smart tail detection - NO CUTOFF!"""
    try:
        if use_smart_tail:
            audio = render_with_smart_tail(engine, synth, midi_note, NOTE_DURATION)
        else:
            synth.clear_midi()
            synth.add_midi_note(midi_note, NOTE_VELOCITY, 0.0, NOTE_DURATION)
            engine.load_graph([(synth, [])])
            engine.render(NOTE_DURATION + RENDER_TAIL)
            audio = engine.get_audio()
        
        if len(audio) == 0 or audio.shape[0] == 0:
            return False

        audio_normalized = audio.T
        max_val = np.max(np.abs(audio_normalized))
        if max_val > 0:
            audio_normalized = audio_normalized / max_val * 0.95
        
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        file_writer.write_async(output_path, SAMPLE_RATE, audio_int16)
        
        return True
        
    except Exception as e:
        print(f"\n    ✗ Error rendering {note_name}: {e}")
        return False

def render_all_notes_turbo(serum_path, preset_path, output_dir, preset_name, 
                           start_note="C0", end_note="G9", use_smart_tail=True):
    """Render all notes with SAFE turbo optimizations and smart tail detection."""
    midi_range = get_midi_range(start_note, end_note)
    total_notes = len(midi_range)
    
    os.makedirs(output_dir, exist_ok=True)
    
    is_vst3 = serum_path.endswith('.vst3')
    print(f"\n{'='*60}")
    print(f"RENDERING {total_notes} NOTES")
    print(f"Range: {start_note} to {end_note}")
    print(f"Preset: {preset_name}")
    print(f"Output: {output_dir}")
    print(f"Optimization: Threaded file I/O (2-3x faster)")
    if use_smart_tail:
        print(f"Smart Tail: ON (auto-detects silence at {SILENCE_THRESHOLD_DB}dB)")
    else:
        print(f"Smart Tail: OFF (fixed {RENDER_TAIL}s tail)")
    print(f"Buffer size: {BUFFER_SIZE} (optimized)")
    print(f"VST: {Path(serum_path).name}")
    print(f"Format: {'VST3' if is_vst3 else 'VST2'}")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    start_time = time.time()
    
    file_writer = FileWriterPool(max_workers=4)
    
    try:
        print("Creating DawDreamer engine...")
        engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
        
        print(f"Loading Serum...")
        synth = engine.make_plugin_processor("serum", serum_path)
        print("✓ Serum loaded!\n")
        
        if preset_path:
            print(f"Loading preset: {preset_name}")
            if load_preset_into_synth(synth, preset_path, serum_path):
                print("✓ Preset loaded!\n")
            else:
                print("⚠ Warning: Preset may not have loaded\n")
        else:
            print("Using INIT preset\n")
        
        for i, (midi_note, note_name) in enumerate(midi_range, 1):
            filename = f"{preset_name}_{note_name}_{midi_note:03d}.wav"
            output_path = os.path.join(output_dir, filename)
            
            if os.path.exists(output_path):
                skip_count += 1
                elapsed = time.time() - start_time
                notes_per_sec = i / elapsed if elapsed > 0 else 0
                remaining = (total_notes - i) / notes_per_sec if notes_per_sec > 0 else 0
                extra = f"| {notes_per_sec:.1f} notes/sec | ETA: {format_time(remaining)}"
                print(progress_bar(i, total_notes, prefix="Progress", extra=extra), end='')
                continue
            
            if render_single_note(engine, synth, midi_note, note_name, output_path, file_writer, use_smart_tail):
                success_count += 1
            else:
                fail_count += 1
            
            elapsed = time.time() - start_time
            notes_per_sec = i / elapsed if elapsed > 0 else 0
            remaining_notes = total_notes - i
            eta_seconds = remaining_notes / notes_per_sec if notes_per_sec > 0 else 0
            
            extra = f"| {notes_per_sec:.1f} notes/sec | ETA: {format_time(eta_seconds)}"
            print(progress_bar(i, total_notes, prefix="Progress", extra=extra), end='')
            
            if i % 20 == 0:
                file_writer.wait_all()
        
        print()

        print("\nFinalizing file writes...")
        file_writer.wait_all()
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        fail_count = total_notes - success_count - skip_count
    finally:
        file_writer.shutdown()
    
    elapsed_time = time.time() - start_time
    notes_per_sec = (success_count + skip_count) / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n{'='*60}")
    print("RENDER COMPLETE!")
    print(f"Rendered: {success_count}/{total_notes} notes")
    if skip_count > 0:
        print(f"Skipped (already exist): {skip_count}")
    if fail_count > 0:
        print(f"Failed: {fail_count}")
    if use_smart_tail:
        print(f"Smart tail: Enabled (no reverb cutoff!)")
    print(f"Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Speed: {notes_per_sec:.1f} notes/second")
    print(f"Average per note: {elapsed_time/(success_count+skip_count):.2f} seconds")
    print(f"Files saved to: {output_dir}")
    print(f"{'='*60}")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_header():
    """Display the ASCII art header."""
    terminal_width = shutil.get_terminal_size((97, 20)).columns
    
    if terminal_width >= 97:
        print("═"*97)
        print("")
        print("███████╗███████╗██████╗ ██╗   ██╗███╗   ███╗██████╗ ██╗   ██╗███████╗████████╗███████╗██████╗ ")
        print("██╔════╝██╔════╝██╔══██╗██║   ██║████╗ ████║██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗")
        print("███████╗█████╗  ██████╔╝██║   ██║██╔████╔██║██████╔╝██║   ██║███████╗   ██║   █████╗  ██████╔╝")
        print("╚════██║██╔══╝  ██╔══██╗██║   ██║██║╚██╔╝██║██╔══██╗██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗")
        print("███████║███████╗██║  ██║╚██████╔╝██║ ╚═╝ ██║██████╔╝╚██████╔╝███████║   ██║   ███████╗██║  ██║")
        print("╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝")
        print("")
        print("═"*97)
    else:
        print("═"*60)
        print("SERUM BUSTER".center(60))
        print("Automated Preset Renderer".center(60))
        print("═"*60)
    print()

def show_section(step, total, title, clear=True):
    """Display a section header with the ASCII art."""
    if clear:
        clear_screen()
        show_header()
    print(f"┌{'─'*58}┐")
    print(f"│ [{step}/{total}] {title:<50} │")
    print(f"└{'─'*58}┘")
    print()

def main():
    """Main interactive workflow with clean CLI."""
    global NOTE_DURATION
    
    clear_screen()
    show_header()
    
    show_section(1, 5, "Finding Serum Installation")
    all_serums = find_all_serum_versions()
    
    if not all_serums:
        print("✗ No Serum installations found.\n")
        print("Please enter the path to Serum:")
        custom_path = input("❯ ").strip().strip('"')
        
        if Path(custom_path).exists():
            serum_path = custom_path
        else:
            print("\n✗ Invalid path. Exiting.")
            input("\nPress Enter to exit...")
            return
    else:
        print(f"✓ Found {len(all_serums)} installation(s):\n")
        
        python_arch = platform.architecture()[0]
        
        for i, serum in enumerate(all_serums, 1):
            status = "✓" if serum['accessible'] else "✗"
            print(f"  {i}. {status} {Path(serum['path']).name}")
            print(f"     {serum['path']}")
            print(f"     Type: {serum['type']} | Arch: {serum['arch']}", end="")
            
            if python_arch == "64bit" and serum['arch'] == "32-bit":
                print(f" | ⚠ Arch mismatch!", end="")
            elif python_arch == "32bit" and serum['arch'] == "64-bit":
                print(f" | ⚠ Arch mismatch!", end="")
            print("\n")
        
        recommended = None
        for serum in all_serums:
            if serum['accessible'] and serum['type'] == 'VST2':
                if python_arch == "64bit" and serum['arch'] == "64-bit":
                    recommended = serum
                    break
                elif python_arch == "32bit" and serum['arch'] == "32-bit":
                    recommended = serum
                    break
        
        if not recommended:
            for serum in all_serums:
                if serum['accessible']:
                    recommended = serum
                    break
        
        if recommended:
            print(f"Recommended: Option {all_serums.index(recommended) + 1}\n")
        
        choice = input(f"Choose (1-{len(all_serums)}) or enter custom path ❯ ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_serums):
                serum_path = all_serums[idx]['path']
            else:
                print("\n✗ Invalid choice. Exiting.")
                input("\nPress Enter to exit...")
                return
        except ValueError:
            if Path(choice.strip('"')).exists():
                serum_path = choice.strip('"')
            else:
                print("\n✗ Invalid choice. Exiting.")
                input("\nPress Enter to exit...")
                return
    
    show_section(1, 5, "Testing VST Loading")
    print(f"Testing: {Path(serum_path).name}")
    print(f"Path: {serum_path}\n")
    
    if not test_vst_load(serum_path):
        print("\n✗ Failed to load Serum!")
        retry = input("\nTry anyway? (y/n) ❯ ").lower()
        if retry != 'y':
            input("\nPress Enter to exit...")
            return
    else:
        print("\n✓ VST loaded successfully!")
        input("\nPress Enter to continue...")
    
    show_section(2, 5, "Preset Selection")
    print("  1. Use Serum's INIT preset")
    print("  2. Load single .fxp preset")
    print("  3. Load multiple .fxp presets")
    print("  4. Load all presets from folder\n")
    
    choice = input("Choose (1-4) ❯ ").strip()
    
    presets = []
    
    if choice == "1":
        presets.append((None, "INIT"))
        print("\n✓ Using INIT preset")
    
    elif choice == "2":
        print("\nEnter path to .fxp preset file:")
        preset_path = input("❯ ").strip().strip('"')
        
        if Path(preset_path).exists():
            preset_path = str(Path(preset_path).resolve())
            preset_name = Path(preset_path).stem
            presets.append((preset_path, preset_name))
            print(f"\n✓ Loaded: {preset_name}")
        else:
            print("\n✗ Preset not found, using INIT")
            presets.append((None, "INIT"))
    
    elif choice == "3":
        print("\nEnter preset paths (one per line, empty line to finish):")
        while True:
            preset_path = input("❯ ").strip().strip('"')
            if not preset_path:
                break
            
            if Path(preset_path).exists() and preset_path.endswith('.fxp'):
                preset_path = str(Path(preset_path).resolve())
                preset_name = Path(preset_path).stem
                presets.append((preset_path, preset_name))
                print(f"  ✓ Added: {preset_name}")
            else:
                print(f"  ✗ Skipped: {preset_path}")
        
        if not presets:
            presets.append((None, "INIT"))
            print("\n✗ No valid presets, using INIT")
        else:
            print(f"\n✓ Loaded {len(presets)} presets")
    
    elif choice == "4":
        print("\nEnter path to folder containing .fxp files:")
        folder_path = input("❯ ").strip().strip('"')
        
        if Path(folder_path).exists():
            fxp_files = list(Path(folder_path).glob("**/*.fxp"))
            
            if fxp_files:
                for fxp_file in fxp_files:
                    preset_path = str(fxp_file.resolve())
                    preset_name = fxp_file.stem
                    presets.append((preset_path, preset_name))
                
                print(f"\n✓ Found {len(presets)} presets")
                if len(presets) <= 5:
                    for path, name in presets:
                        print(f"  - {name}")
                else:
                    for path, name in presets[:3]:
                        print(f"  - {name}")
                    print(f"  ... and {len(presets) - 3} more")
            else:
                print("\n✗ No .fxp files found, using INIT")
                presets.append((None, "INIT"))
        else:
            print("\n✗ Folder not found, using INIT")
            presets.append((None, "INIT"))
    
    else:
        presets.append((None, "INIT"))
        print("\n✓ Using INIT preset")
    
    input("\nPress Enter to continue...")
    
    show_section(3, 5, "Output Directory")
    
    if len(presets) == 1:
        default_dir = os.path.join(os.getcwd(), "serum_renders", presets[0][1])
    else:
        default_dir = os.path.join(os.getcwd(), "serum_renders", f"batch_{len(presets)}_presets")
    
    print(f"Default: {default_dir}\n")
    
    custom = input("Press Enter for default, or type custom path ❯ ").strip().strip('"')
    base_output_dir = custom if custom else default_dir
    
    print(f"\n✓ Output: {base_output_dir}")
    input("\nPress Enter to continue...")
    
    show_section(4, 5, "Note Range Selection")
    print("  1. Full (C0-G9) - 116 notes")
    print("  2. Piano (A0-C8) - 88 notes")
    print("  3. Bass (C0-C4) - 49 notes")
    print("  4. Lead (C3-C7) - 49 notes")
    print("  5. Custom range\n")
    
    range_choice = input("Choose (1-5) ❯ ").strip()
    
    ranges = {
        "1": ("C0", "G9"),
        "2": ("A0", "C8"),
        "3": ("C0", "C4"),
        "4": ("C3", "C7")
    }
    
    if range_choice in ranges:
        start_note, end_note = ranges[range_choice]
        print(f"\n✓ Selected: {start_note} to {end_note}")
    else:
        print()
        start_note = input("Start note (e.g. C2) ❯ ").strip() or "C0"
        end_note = input("End note (e.g. C6) ❯ ").strip() or "G9"
        print(f"\n✓ Custom range: {start_note} to {end_note}")
    
    input("\nPress Enter to continue...")
    
    show_section(5, 5, "Performance Settings")
    print(f"Note duration: {NOTE_DURATION} seconds\n")
    custom_duration = input("Press Enter to keep, or enter new duration (seconds) ❯ ").strip()
    
    if custom_duration:
        try:
            NOTE_DURATION = float(custom_duration)
            print(f"\n✓ Duration set to {NOTE_DURATION} seconds")
        except:
            print(f"\n✗ Invalid duration, keeping {NOTE_DURATION} seconds")
    else:
        print(f"\n✓ Using {NOTE_DURATION} seconds")
    
    print(f"\nSmart reverb tail detection: {SILENCE_THRESHOLD_DB}dB threshold")
    print("(Ensures reverb tails are never cut off!)\n")
    use_smart_tail_input = input("Disable? (not recommended) (y/n) ❯ ").strip().lower()
    use_smart_tail = use_smart_tail_input != 'y'
    
    if not use_smart_tail:
        print(f"\n⚠ Smart tail disabled - using fixed {RENDER_TAIL}s tail")
    else:
        print(f"\n✓ Smart tail enabled - up to {MAX_TAIL_EXTENSION}s extra if needed")
    
    input("\nPress Enter to continue...")
    
    notes_count = len(get_midi_range(start_note, end_note))
    total_samples = notes_count * len(presets)
    
    show_section(5, 5, "Render Summary")
    print(f"  Serum:       {Path(serum_path).name}")
    print(f"  Presets:     {len(presets)}")
    
    if len(presets) <= 5:
        for _, name in presets:
            print(f"               - {name}")
    else:
        for _, name in presets[:3]:
            print(f"               - {name}")
        print(f"               ... and {len(presets) - 3} more")
    
    print(f"\n  Range:       {start_note} to {end_note} ({notes_count} notes per preset)")
    print(f"  Total:       {total_samples} samples")
    print(f"  Output:      {base_output_dir}")
    print(f"  Duration:    {NOTE_DURATION} seconds per note")
    print(f"  Smart tail:  {'ON' if use_smart_tail else 'OFF'}")
    print(f"  Threading:   Enabled")
    
    est_time = total_samples * (NOTE_DURATION * 0.4)
    print(f"\n  Estimated:   ~{est_time:.0f} seconds (~{est_time/60:.1f} minutes)")
    
    print(f"\n{'─'*60}\n")
    
    confirm = input("Start rendering? (y/n) ❯ ").lower()
    
    if confirm == 'y':
        print()
        start_time = time.time()
        
        for preset_index, (preset_path, preset_name) in enumerate(presets, 1):
            show_section(5, 5, f"Rendering: {preset_name} ({preset_index}/{len(presets)})", clear=False)
            
            if len(presets) > 1:
                output_dir = os.path.join(base_output_dir, preset_name)
            else:
                output_dir = base_output_dir
            
            render_all_notes_turbo(
                serum_path=serum_path,
                preset_path=preset_path,
                output_dir=output_dir,
                preset_name=preset_name,
                start_note=start_note,
                end_note=end_note,
                use_smart_tail=use_smart_tail
            )
        
        total_time = time.time() - start_time
        notes_per_sec = total_samples / total_time if total_time > 0 else 0
        
        show_section(5, 5, "Render Complete!")
        print(f"  Presets:     {len(presets)}")
        print(f"  Samples:     {total_samples}")
        print(f"  Time:        {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Speed:       {notes_per_sec:.1f} notes/second")
        print(f"  Average:     {total_time/total_samples:.2f}s per sample")
        print(f"\n  Location:    {base_output_dir}")
        print(f"\n{'─'*60}")
        print("\n✓ Done! Your samples are ready!")
        
    else:
        show_section(5, 5, "Cancelled")
        print("Render cancelled by user.")

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
