#!/usr/bin/env python3
"""
Main entry point for BPM Analyzer using PySide6
"""

import sys
from PySide6.QtWidgets import QApplication
from analyzer import BPMGUIApp


def main():
    """Main function to launch the BPM Analyzer application"""
    app = QApplication(sys.argv)
    
    # Set application name and style
    app.setApplicationName("Advanced BPM Analyzer")
    
    # Create and show the main window
    window = BPMGUIApp()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
