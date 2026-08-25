# Dual Camera Footage Studio

A locally-hosted web app for processing dual action camera footage into single panoramic videos covering the entire pitch. Initially designed for sports teams recording their games and training sessions.

Footage Studio takes the raw clips from two cameras mounted side-by-side on a tripod, groups them into recording sessions, and stitches the left and right camera footage together into a single wide panoramic output — giving you a hands-free, full-pitch recording with no subscription fees and no ongoing costs.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Footage recorded from two devices (same model) with overlapping FoV
- A computer or external hard drive with enough storage for your footage

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/dual-camera-footage-studio.git
cd dual-camera-footage-studio
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Set up your footage directory

On your storage device (or computer), create a folder for your footage with the following structure:

```
Footage/
├── Left Camera/
├── Right Camera/
└── Output Footage/
```

### 4. Start the app

```bash
uv run footage-studio
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### 5. Configure settings

On the Home page, enter the full path to your footage folder (e.g. `/Volumes/MyDrive/Team Footage`) and click Save.

---

## Standard Workflow

### After filming

1. Transfer your footage from your recording devices to your storage device — left camera clips into `Left Camera/`, right camera clips into `Right Camera/`. Whilst the stitching together of videos doesn't require you to know which side the footage comes from, it makes the next step easier.

2. Open the app and navigate to the **Group** page. Click **Scan** to detect your recordings. The app will group consecutive clips from the same recording session together. Review the groups and click **Group** to concatenate each group into a single file. This step is neccesary for any recording devices that record onto an SD card, as the file formatting only allows files up to a size of 4GB and so will split longer recordings into chunks that are <= 4GB in size.

3. Navigate to the **Stitch** page. Click **Scan** to match up your left and right camera sessions. Give each session a name (e.g. "Training 2026-01-01" or "Team 1 Vs Team 2 2026-01-01"), untick any sessions you don't want to process, then click **Stitch Selected**. This step can take quite a while depending on the size of the files and what device you are using to do the processing.

4. Navigate to the **Browse** page to view your finished panoramic output videos.

---

## Hardware Setup

For best results, mount both cameras on a dual camera mount attached to a tripod or mast positioned at the side of the pitch, roughly 3–4 metres high. Each camera should cover one half of the pitch with slight overlap in the centre.
