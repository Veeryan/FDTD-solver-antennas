# App Icon and Assets

Place your app icon here so the GUI shows a custom icon (leaf) instead of the default Python icon.

## Supported icon locations (checked in this order)
- assets/app.ico
- assets/leaf.ico
- icon.ico (project root)

If no .ico is found, the app looks for a PNG fallback:
- assets/app.png
- assets/leaf.png

On Windows, `.ico` is strongly recommended for reliable taskbar/titlebar icons.

## Recommended .ico contents
- Provide a single .ico file containing multiple sizes for best DPI scaling:
  - 16, 20, 24, 32, 48, 64, 128, 256 px
- Color depth: 32-bit RGBA with transparency.
- A clean, square canvas (1:1 aspect).

Tools to create .ico from a PNG:
- GIMP, Photoshop, Inkscape (export as ICO)
- Online converters (e.g. search for "PNG to ICO multi-size")

## Where this is used in the code
The icon is loaded in `gui_app.py` inside `AntennaSimulatorGUI.setup_window()`:
- Tries ICO first: `assets/app.ico`, `assets/leaf.ico`, `icon.ico`
- Falls back to PNG via `iconphoto` if available

Windows-specific setup:
- The app sets a Windows AppUserModelID (`VeeryAntenna.PatchDesigner.GUI`) and enables per-monitor DPI awareness so the icon groups correctly in the taskbar and scales well on HiDPI displays.

## If the icon doesn’t update
- Close all running instances of the app and start it again.
- Windows sometimes caches icons per AppUserModelID. If it still shows the Python icon, try:
  - Sign out and back in, or
  - Clear the Windows icon cache (search online for steps), or
  - Temporarily rename the ICO file (e.g. `leaf_v2.ico`) and replace it.

## Additional assets
You can store any other app images in this `assets/` folder. Keep filenames simple (no spaces or special characters).
