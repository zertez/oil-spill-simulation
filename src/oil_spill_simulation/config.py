import os

import toml


class TOMLFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = None  # Initialize the config attribute

    def read_config(self):
        if not os.path.exists(self.file_path):
            # Raise an error if the file path does not exist
            raise FileNotFoundError(f"The specified config file does not exist: {self.file_path}")

        # Load the configuration from the TOML file
        self.config = toml.load(self.file_path)
        # Validate the configuration to ensure required sections and settings are present
        self.validate_config()
        # Set default values for missing settings
        self.set_defaults()

    def validate_config(self):
        # Define required sections in the configuration
        required_sections = ["settings", "geometry", "IO"]
        for section in required_sections:
            if section not in self.config:
                # Raise an error if a required section is missing
                raise ValueError(f"Missing required section: {section}")

        # Define required settings within the 'settings' section
        required_settings = ["nSteps", "tEnd"]
        for setting in required_settings:
            if setting not in self.config["settings"]:
                # Raise an error if a required setting is missing
                raise ValueError(f"Missing required setting: {setting}")

        # Check for required parameters in the 'geometry' section
        if "meshName" not in self.config["geometry"]:
            raise ValueError("Missing required geometry parameter: meshName")

        if "borders" not in self.config["geometry"]:
            raise ValueError("Missing required geometry parameter: borders")

        # Validate restart file and start time consistency
        self.validate_restart_configuration()

    def set_defaults(self):
        # Set default values for optional settings if they are not provided
        self.config["settings"].setdefault("tStart", 0.0)
        self.config["IO"].setdefault("logName", "logfile.mp4")  # Default log file with a valid extension
        self.config["IO"].setdefault("writeFrequency", None)
        self.config["IO"].setdefault("restartFile", None)

        # Ensure the logName has a valid video extension
        if not self.config["IO"]["logName"].endswith((".mp4", ".avi", ".mkv")):
            self.config["IO"]["logName"] += ".mp4"

    def validate_restart_configuration(self):
        """Validate restart file and start time consistency."""
        restart_file = self.config["IO"].get("restartFile")
        t_start = self.config["settings"].get("tStart")

        if restart_file is not None and restart_file != "":
            # If restart file is provided, start time must be provided and > 0
            if t_start is None or t_start <= 0:
                raise ValueError("When restartFile is provided, tStart must be provided and greater than 0")

            # Check if restart file exists
            if not os.path.exists(restart_file):
                raise FileNotFoundError(f"Restart file not found: {restart_file}")

        # Note: It's valid to have tStart > 0 without a restart file
        # This just means starting the simulation from that time with initial conditions

    def get_config(self):
        if self.config is None:
            # Read the config if it hasn't been read yet
            self.read_config()
        return self.config
