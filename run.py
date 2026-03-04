#!/usr/bin/env python3
"""
Entry point for the AI Skin Disease Detection System.
Launches the Streamlit frontend.
"""

import subprocess
import sys
import os


def main() -> None:
    """Launch the Streamlit application."""
    app_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")

    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        sys.exit(1)

    print("🔬 Starting AI Skin Disease Detection System...")
    print(f"   App: {app_path}")
    print("   Press Ctrl+C to stop\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_path,
             "--server.headless", "true"],
            cwd=os.path.dirname(__file__),
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped.")
    except FileNotFoundError:
        print("Error: streamlit is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
