<a href="#serumbuster"><img width="150" alt="sb" src="https://github.com/user-attachments/assets/de8acae3-674e-47fc-8b7e-c4b73c0a411d" /></a> <a href="#fastwasher"><img width="150" alt="FW" src="https://github.com/user-attachments/assets/6321c0da-dd6d-4644-b0b4-87150b70cd83" /></a> <a href="#fastplugger"><img width="150" alt="FP" src="https://github.com/user-attachments/assets/57ed4110-aee5-49ee-81fe-e187286e51b1" /></a>


# SerumBuster Toolkit

A comprehensive toolkit for working with Xfer Serum presets and FastTracker 2 instruments. SerumBuster provides three specialized tools for rendering Serum presets to audio and converting sample libraries into FastTracker 2 XI format.

## Overview

SerumBuster consists of three Python tools designed to streamline the workflow between modern synthesizers and classic tracker formats:

- **SerumBuster** - Batch render Serum FXP presets to individual WAV files
- **FastWasher** - Convert Serum FXP presets directly to FastTracker 2 XI instruments
- **FastPlugger** - Convert WAV keymap folders to FastTracker 2 XI instruments

## Table of Contents

- [Installation](#installation)
- [System Requirements](#system-requirements)
- [SerumBuster](#serumbuster)
- [FastWasher](#fastwasher)
- [FastPlugger](#fastplugger)
- [Usage](#usage)
  - [SerumBuster Usage](#serumbuster-usage)
  - [FastWasher Usage](#fastwasher-usage)
  - [FastPlugger Usage](#fastplugger-usage)
- [Technical Details](#technical-details)
- [File Formats](#file-formats)
- [Workflow Examples](#workflow-examples)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)

## Installation

### Prerequisites

All three tools require Python 3.7 or higher and several dependencies:

```bash
pip install dawdreamer numpy scipy
```

### Dependency Details

- **dawdreamer** - VST plugin hosting and rendering engine
- **numpy** - Numerical processing for audio manipulation
- **scipy** - Audio file I/O and signal processing

### Installation Steps

1. Ensure Python 3.7+ is installed
2. Install required dependencies using pip
3. Download the SerumBuster toolkit scripts
4. Ensure Xfer Serum is installed (for SerumBuster and FastWasher)

## System Requirements

### Operating System
- Windows (primary support)
- macOS and Linux (experimental)

### Hardware
- Multi-core processor recommended for batch processing
- Minimum 4GB RAM
- Sufficient disk space for rendered samples

### Software
- Xfer Serum VST2 or VST3 (for SerumBuster and FastWasher)
- Python 3.7 or higher
- FastTracker 2, MilkyTracker, or compatible tracker software (for using XI files)

---

<img width="800" height="360" alt="sb-banner" src="https://github.com/user-attachments/assets/b2e6eed9-8de6-47f3-8cdd-85a21457cde3" />

## SerumBuster

SerumBuster is the core rendering tool that converts Serum FXP presets into individual WAV files across a specified MIDI note range.

#### Key Features

- Batch rendering of multiple Serum presets
- Intelligent reverb tail detection to prevent cut-off effects
- Multi-threaded file writing for improved performance
- Customizable note ranges and durations
- Support for both single presets and bulk folder processing
- Automatic Serum VST detection

#### Use Cases

- Creating sample libraries from Serum presets
- Generating individual notes for use in other samplers
- Archiving Serum presets as audio
- Preparing samples for further processing

---

<img width="800" height="179" alt="fastwasher" src="https://github.com/user-attachments/assets/d5f1346b-1d41-49b6-8e51-5c6439bcd766" />

## FastWasher

FastWasher combines the functionality of SerumBuster with automatic XI conversion, providing a direct path from Serum presets to FastTracker 2 instruments.

#### Key Features

- Single-step conversion from FXP to XI format
- Optimal 16-sample distribution for FastTracker 2 compatibility
- Batch processing with folder structure mirroring
- Integrated sample rendering and instrument building
- Smart sample selection algorithm
- Automatic keymap generation

#### Use Cases

- Creating tracker instruments directly from Serum presets
- Converting entire preset libraries to XI format
- Rapid instrument prototyping for tracker compositions
- Building custom instrument banks

---

<img width="800" height="179" alt="FastPlugger" src="https://github.com/user-attachments/assets/a8363a47-ec1f-4362-ae24-34a30f59b296" />

## FastPlugger

FastPlugger converts existing WAV file collections into FastTracker 2 XI instruments, ideal for organizing pre-rendered samples or external sample libraries.

#### Key Features

- Automatic note detection from filenames
- Support for custom naming conventions
- Bulk processing of multiple sample folders
- Interactive and command-line modes
- Intelligent sample selection (up to 16 samples)
- Preservation of original sample quality

#### Use Cases

- Converting existing sample libraries to XI format
- Organizing loose sample collections into instruments
- Creating instruments from field recordings
- Importing samples from other samplers

## Usage

### SerumBuster Usage

#### Interactive Mode

Run the script without arguments for the interactive wizard:

```bash
python SerumBuster.py
```

#### Interactive Workflow

1. **Serum VST Detection**
   - Tool automatically scans for Serum installations
   - Select from detected versions if multiple found
   - Verifies VST can be loaded successfully

2. **Preset Selection**
   - Option 1: Use INIT preset (default Serum sound)
   - Option 2: Load single FXP file
   - Option 3: Load multiple specific FXP files
   - Option 4: Load all FXP files from folder (recursive)

3. **Output Directory**
   - Default: `./serum_renders/[preset_name]`
   - Custom path can be specified
   - Batch renders create subdirectories per preset

4. **Note Range Selection**
   - Full (C0-G9): 116 notes - complete range
   - Piano (A0-C8): 88 notes - standard piano range
   - Bass (C0-C4): 49 notes - low frequency focused
   - Lead (C3-C7): 49 notes - mid-to-high range
   - Custom: User-defined start and end notes

5. **Performance Settings**
   - Note duration: How long each note is held (default: 4.0 seconds)
   - Smart tail detection: Automatically extends render time for reverb/delay tails
   - Threading: Parallel file writing for faster processing

#### Output

SerumBuster creates a folder structure:
```
output_directory/
├── preset_name/
│   ├── preset_name_C0_0_24.wav
│   ├── preset_name_C#0_1_25.wav
│   ├── preset_name_D0_2_26.wav
│   └── ...
```

Each WAV file is named: `[preset]_[note]_[octave]_[midi].wav`

### FastWasher Usage

#### Interactive Mode

Run the script for the step-by-step interface:

```bash
python FastWasher.py
```

#### Workflow Steps

1. **Serum VST Detection**
   - Automatic scanning and selection
   - VST load verification

2. **Processing Mode**
   - Single FXP: Convert one preset to XI
   - Bulk: Process entire folder of presets

3. **Note Range Selection**
   - Same ranges as SerumBuster
   - FastWasher automatically selects 16 optimally distributed samples
   - Range contains fewer than 16 notes: Uses all available notes
   - Range contains more than 16 notes: Evenly distributes across range

4. **Processing**
   - Renders samples using SerumBuster engine
   - Automatically converts to XI using FastPlugger engine
   - Creates XI file ready for FastTracker 2

#### Output

Single mode creates:
```
output_directory/
└── preset_name.xi
```

Bulk mode mirrors folder structure:
```
output_directory/
├── subfolder1/
│   ├── preset1.xi
│   └── preset2.xi
└── subfolder2/
    └── preset3.xi
```

### FastPlugger Usage

#### Interactive Mode

```bash
python FastPlugger.py
```

#### Interactive Workflow

1. **Input Directory Selection**
   - Specify folder containing WAV files
   - Tool scans for subdirectories with WAV files
   - Choose single folder or bulk processing

2. **Output Configuration**
   - Specify output XI filename
   - Choose instrument name (max 22 characters)
   - Confirm overwrite if file exists

3. **Processing**
   - Automatic note detection from filenames
   - Sample loading and conversion
   - XI file generation

#### Command-Line Mode

For automation and scripting:

```bash
python FastPlugger.py <input_dir> <output_file> [options]
```

**Arguments:**
- `input_dir`: Directory containing WAV files
- `output_file`: Output XI file path

**Options:**
- `-n, --name`: Instrument name (default: "Instrument")
- `-m, --max-samples`: Maximum samples to include (default: 16)
- `-l, --max-length`: Maximum sample length in frames (0 = keep full)
- `-v, --verbose`: Show detailed processing information

**Example:**
```bash
python FastPlugger.py ./my_samples output.xi -n "My Synth" -m 16 -v
```

#### WAV File Naming Convention

FastPlugger expects WAV files to follow this naming pattern:

```
[name]_[note][octave]_[midi].wav
```

Examples:
- `synth_C4_60.wav`
- `bass_A#2_46.wav`
- `lead_G5_79.wav`

Components:
- `[name]`: Any descriptive text
- `[note]`: Musical note (C, C#, D, etc.)
- `[octave]`: Octave number (0-9)
- `[midi]`: MIDI note number (0-127)

## Technical Details

### Audio Processing

#### Sample Rate
All tools operate at 44.1kHz, the standard sample rate for FastTracker 2 and most tracker software.

#### Bit Depth
Samples are processed as 16-bit signed integers, matching the FastTracker 2 XI format specification.

#### Format Support
- Input: WAV files (8-bit, 16-bit, 24-bit, 32-bit float)
- Output: 16-bit mono WAV and XI format
- Stereo files are automatically downmixed to mono

### Smart Tail Detection

FastWasher and SerumBuster implement intelligent tail detection to prevent reverb and delay effects from being cut off:

- Silence threshold: -80dB
- Check window: 100ms
- Maximum tail extension: 10 seconds
- Algorithm monitors audio output after note release
- Automatically extends render time if signal remains above threshold

### Sample Selection Algorithm

When more than 16 samples are available, FastPlugger and FastWasher use an intelligent selection algorithm:

1. Parse all sample filenames for MIDI note numbers
2. Sort samples by MIDI note value
3. Calculate even distribution across available range
4. Select 16 samples that provide optimal keyboard coverage
5. Generate keymap assigning MIDI ranges to selected samples

### Threading and Performance

SerumBuster and FastWasher use multi-threaded file writing:

- Main thread handles VST rendering (sequential)
- Worker thread pool (4 threads) handles file I/O
- Reduces total render time by parallelizing disk operations
- Prevents I/O bottlenecks during batch processing

### VST Loading

The tools search for Serum in these locations (Windows):

```
C:\Program Files\VSTPlugins\Xfer\Serum\Serum_x64.dll
C:\Program Files\VSTPlugins\Serum_x64.dll
C:\Program Files\VSTPlugins\Serum.dll
C:\Program Files\Steinberg\VSTPlugins\Serum.dll
C:\Program Files\Steinberg\VSTPlugins\Serum_x64.dll
C:\Program Files\Common Files\VST2\Serum.dll
C:\Program Files\Common Files\VST2\Serum_x64.dll
C:\Program Files (x86)\VSTPlugins\Serum.dll
C:\Program Files (x86)\Steinberg\VSTPlugins\Serum.dll
C:\Program Files\Common Files\VST3\Serum.vst3
```

Additional recursive scanning is performed in:
- Program Files directories
- Common VST plugin folders

## File Formats

### XI Format Specification

The FastTracker 2 Extended Instrument (XI) format:

**Header Structure:**
- Extended Instrument identifier (60 bytes)
- Instrument name (22 characters, padded)
- Tracker name (20 bytes)
- Version number (2 bytes)
- Number of samples (2 bytes)
- Sample header size (4 bytes)

**Sample Data:**
- Note mapping table (96 entries)
- Volume envelope points
- Panning envelope points
- Volume/panning envelope parameters
- Vibrato settings
- Fadeout value
- Reserved space

**Per-Sample Headers:**
- Sample length (4 bytes)
- Loop start (4 bytes)
- Loop end (4 bytes)
- Volume (1 byte)
- Finetune (1 byte, signed)
- Sample type flags (1 byte)
- Panning (1 byte)
- Relative note (1 byte, signed)
- Reserved (1 byte)
- Sample name (22 bytes)

**Sample Audio Data:**
- 16-bit signed PCM samples
- Delta-encoded for compression
- Mono channel

### FXP Format

Serum FXP files are VST preset containers:
- Binary format defined by Steinberg VST specification
- Contains parameter values and preset name
- Loaded by VST host (DawDreamer) and applied to plugin

### WAV Format

Standard RIFF WAVE format:
- RIFF header with WAVE identifier
- Format chunk (sample rate, bit depth, channels)
- Data chunk (raw audio samples)
- Optional metadata chunks

## Workflow Examples

### Example 1: Converting a Single Serum Preset to XI

```
1. Run FastWasher.py
2. Select detected Serum installation
3. Choose "Single FXP file"
4. Browse to preset: "C:\Presets\MyBass.fxp"
5. Set output: "C:\Output\MyBass.xi"
6. Select range: Bass (C0-C4)
7. Confirm and render
8. Load MyBass.xi in FastTracker 2
```

### Example 2: Batch Converting Preset Library

```
1. Run FastWasher.py
2. Select Serum installation
3. Choose "Bulk process"
4. Input folder: "C:\Serum\Presets\Bass"
5. Output folder: "C:\XI_Library\Bass"
6. Select range: Bass (C0-C4)
7. Process all presets (folder structure maintained)
8. Import entire XI_Library folder into tracker
```

### Example 3: Creating Sample Library from Serum

```
1. Run SerumBuster.py
2. Select Serum installation
3. Choose "Load all presets from folder"
4. Input folder: "C:\Serum\Presets\Pads"
5. Output: "C:\Samples\Pad_Library"
6. Select range: Full (C0-G9)
7. Duration: 6.0 seconds (longer for pads)
8. Enable smart tail detection
9. Render creates organized WAV library
10. Use samples in any DAW or sampler
```

### Example 4: Converting Existing Sample Collection

```
1. Organize WAV files with proper naming:
   - brass_C3_48.wav
   - brass_E3_52.wav
   - brass_G3_55.wav
   - etc.
2. Run FastPlugger.py
3. Select input directory: "C:\Samples\Brass"
4. Output: "C:\XI\Brass_Section.xi"
5. Instrument name: "Brass Section"
6. Load in tracker and play
```

### Example 5: Complete Production Workflow

```
1. Design sounds in Serum
2. Export presets as FXP files
3. Use FastWasher for instant XI conversion
4. Load XI instruments in MilkyTracker
5. Compose tracker module
6. Export final song as WAV/MP3

Alternative path:
1. Use SerumBuster for maximum flexibility
2. Process WAV files in audio editor
3. Organize processed samples
4. Use FastPlugger to create final XI
5. Import to tracker
```

## Troubleshooting

### Serum Not Detected

**Problem:** Tool cannot find Serum installation

**Solutions:**
- Ensure Serum is installed in standard VST directory
- Check that Serum DLL is not locked by another application
- Verify Serum architecture matches Python (64-bit recommended)
- Try running as administrator if file access is denied
- Check file permissions on Serum installation directory

### VST Load Failure

**Problem:** "Failed to load" error when testing VST

**Solutions:**
- Verify DawDreamer installation: `pip install --upgrade dawdreamer`
- Check Python architecture matches Serum (both 64-bit or both 32-bit)
- Ensure all Serum dependencies are installed
- Try different Serum version (VST2 vs VST3)
- Temporarily disable antivirus during load

### No WAV Files Found

**Problem:** FastPlugger reports no WAV files in directory

**Solutions:**
- Verify files have .wav or .WAV extension
- Check file naming follows convention: `name_NOTE#_MIDI.wav`
- Ensure files are not in nested subdirectories (unless using bulk mode)
- Check file permissions allow reading

### Incorrect Note Detection

**Problem:** Samples mapped to wrong keys

**Solutions:**
- Verify filename format: `[name]_[note][octave]_[midi].wav`
- Check note names use proper format: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- Ensure octave number is single digit (0-9)
- Verify MIDI number matches note/octave
- Test with verbose mode: `python FastPlugger.py input/ output.xi -v`

### XI File Won't Load in Tracker

**Problem:** Generated XI file not recognized

**Solutions:**
- Verify target tracker supports XI format (FastTracker 2, MilkyTracker, etc.)
- Check file size is reasonable (not 0 bytes)
- Try loading in different tracker software
- Regenerate XI with verbose output to check for errors
- Ensure source samples are valid audio files

### Samples Cut Off Prematurely

**Problem:** Reverb tails or sustained notes truncated

**Solutions:**
- Ensure smart tail detection is enabled (FastWasher/SerumBuster)
- Increase note duration: 6-8 seconds for pads/ambient sounds
- Check silence threshold is appropriate (-80dB default)
- Manually extend max tail extension if needed
- Verify preset doesn't have excessively long release/reverb

### Memory Errors During Batch Processing

**Problem:** Out of memory errors with large preset collections

**Solutions:**
- Process smaller batches of presets
- Reduce note range (fewer samples per preset)
- Close other applications to free RAM
- Reduce note duration if applicable
- Use 64-bit Python installation

### Slow Processing Speed

**Problem:** Rendering takes excessive time

**Solutions:**
- Ensure multi-threading is enabled (default)
- Reduce note range if full range not needed
- Decrease note duration for percussive sounds
- Close other applications to free CPU
- Disable smart tail detection for sounds without tails
- Process in smaller batches

### File Permission Errors

**Problem:** Cannot write output files

**Solutions:**
- Check output directory exists and is writable
- Run with administrator privileges if needed
- Ensure output path is not in protected system directory
- Verify disk has sufficient free space
- Close files if they're open in other applications

## Known Limitations

### FastTracker 2 Limitations

- Maximum 16 samples per instrument
- Mono samples only (stereo downmixed)
- 16-bit sample resolution
- No built-in effects in XI format
- Sample names limited to 22 characters

### Tool Limitations

- Windows is primary supported platform (macOS/Linux experimental)
- Requires Serum installation for SerumBuster and FastWasher
- VST2/VST3 only (no AU or other formats)
- No real-time preview during rendering
- Cannot extract from Serum's native .fxp files (requires VST rendering)
- Limited to 44.1kHz sample rate
- No batch resume/checkpoint system for interrupted renders

### Format Limitations

- XI format has no standardized compression
- Large sample libraries result in large XI files
- No support for stereo samples
- Loop points must be manually configured if needed
- No metadata preservation beyond sample names

### Performance Limitations

- VST rendering is sequential (cannot parallel render same instance)
- File I/O threading limited to 4 workers
- Large sample counts increase processing time significantly
- Memory usage scales with sample count and length

## Best Practices

### Naming Conventions

- Use descriptive, consistent instrument names
- Follow WAV naming pattern strictly for FastPlugger
- Keep preset names under 22 characters for XI compatibility
- Use underscores instead of spaces in filenames

### Sample Quality

- Use appropriate note ranges for instrument type
- Choose longer durations for sustained sounds
- Enable smart tail for reverb/delay-heavy presets
- Normalize samples if needed before XI conversion

### Organization

- Maintain organized folder structure for presets
- Use bulk processing for entire libraries
- Keep source FXP files separate from rendered output
- Create separate XI libraries by instrument category

### Performance

- Process in batches to manage memory usage
- Use appropriate note ranges to minimize sample count
- Disable smart tail for percussive sounds
- Close unnecessary applications during rendering

### Quality Control

- Test XI files in tracker before committing to batch
- Verify sample mapping across keyboard range
- Check for clipping or distortion in rendered samples
- Ensure reverb tails are not cut off

## License and Credits

SerumBuster toolkit created for the tracker music community.

Third-party dependencies:
- DawDreamer - VST hosting library
- NumPy - Numerical computation
- SciPy - Scientific computing and audio I/O

Xfer Serum is a product of Xfer Records.
FastTracker 2 is a product of Triton/Starbreeze.

## Version History

Current version includes:
- SerumBuster core rendering engine
- FastWasher FXP to XI converter
- FastPlugger WAV to XI converter
- Interactive and command-line interfaces
- Multi-threaded performance optimization
- Smart tail detection system
- Bulk processing capabilities
- Automatic VST detection

## Support

For issues, suggestions, or contributions, please ensure you:
- Have met all system requirements
- Have installed all required dependencies
- Have verified Serum installation and licensing
- Have checked the troubleshooting section

Common issues can usually be resolved by:
- Updating dependencies to latest versions
- Verifying file paths and permissions
- Checking Serum VST architecture compatibility
- Reviewing the documentation thoroughly
