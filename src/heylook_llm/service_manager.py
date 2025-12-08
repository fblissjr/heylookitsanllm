# src/heylook_llm/service_manager.py
"""
Service management for heylookllm - runs server as background service on macOS and Linux.

Security defaults:
- Binds to 127.0.0.1 (localhost only) by default
- For LAN access behind VPN, use --host 0.0.0.0 with appropriate firewall rules

Supported platforms:
- macOS: launchd (LaunchAgents for user-level service)
- Linux: systemd (user-level or system-level service)
"""

import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_platform() -> str:
    """Detect the current platform."""
    if sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    else:
        return "unsupported"


def get_templates_dir() -> Path:
    """Get the path to service templates."""
    # Templates are in the services/ directory at the repo root
    module_dir = Path(__file__).parent
    # Go up from src/heylook_llm to repo root
    repo_root = module_dir.parent.parent
    templates_dir = repo_root / "services"

    if not templates_dir.exists():
        # Fallback: check relative to current working directory
        templates_dir = Path.cwd() / "services"

    return templates_dir


def detect_venv_path() -> Path:
    """Detect the virtual environment path."""
    # Check if we're in a venv
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)

    # Check for .venv in current directory
    cwd_venv = Path.cwd() / ".venv"
    if cwd_venv.exists():
        return cwd_venv

    # Fallback to sys.prefix
    return Path(sys.prefix)


def install_service_linux(
    host: str = "127.0.0.1",
    port: int = 8080,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    system_wide: bool = False,
) -> bool:
    """Install systemd service on Linux.

    Args:
        host: Host to bind to (127.0.0.1 for local only, 0.0.0.0 for LAN)
        port: Port to run the server on
        log_level: Logging level
        log_dir: Directory for log files
        system_wide: Install as system service (requires sudo) vs user service

    Returns:
        True if installation succeeded
    """
    templates_dir = get_templates_dir()
    template_path = templates_dir / "heylookllm.service.template"

    if not template_path.exists():
        logger.error(f"Service template not found: {template_path}")
        return False

    # Read template
    template = template_path.read_text()

    # Prepare variables
    working_dir = Path.cwd()
    venv_path = detect_venv_path()
    user = os.environ.get("USER", os.environ.get("LOGNAME", "root"))
    group = user  # Usually same as user on Linux

    if log_dir is None:
        log_dir = str(working_dir / "logs")

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Replace placeholders
    service_content = template.format(
        USER=user,
        GROUP=group,
        WORKING_DIR=str(working_dir),
        VENV_PATH=str(venv_path),
        HOST=host,
        PORT=str(port),
        LOG_LEVEL=log_level,
        LOG_DIR=log_dir,
    )

    # Determine installation path
    if system_wide:
        service_path = Path("/etc/systemd/system/heylookllm.service")
        if os.geteuid() != 0:
            logger.error("System-wide installation requires root privileges. Use sudo.")
            return False
    else:
        # User-level service
        user_systemd = Path.home() / ".config" / "systemd" / "user"
        user_systemd.mkdir(parents=True, exist_ok=True)
        service_path = user_systemd / "heylookllm.service"

    # Write service file
    try:
        service_path.write_text(service_content)
        logger.info(f"Service file written to: {service_path}")
    except PermissionError:
        logger.error(f"Permission denied writing to {service_path}")
        return False

    # Reload systemd and enable service
    try:
        if system_wide:
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", "heylookllm.service"], check=True)
        else:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "enable", "heylookllm.service"], check=True)
            # Enable lingering so user services persist after logout
            subprocess.run(["loginctl", "enable-linger", user], check=False)

        logger.info("Service enabled successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to enable service: {e}")
        return False


def install_service_macos(
    host: str = "127.0.0.1",
    port: int = 8080,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
) -> bool:
    """Install launchd service on macOS.

    Args:
        host: Host to bind to (127.0.0.1 for local only, 0.0.0.0 for LAN)
        port: Port to run the server on
        log_level: Logging level
        log_dir: Directory for log files

    Returns:
        True if installation succeeded
    """
    templates_dir = get_templates_dir()
    template_path = templates_dir / "com.heylookllm.server.plist.template"

    if not template_path.exists():
        logger.error(f"Service template not found: {template_path}")
        return False

    # Read template
    template = template_path.read_text()

    # Prepare variables
    working_dir = Path.cwd()
    venv_path = detect_venv_path()

    if log_dir is None:
        log_dir = str(working_dir / "logs")

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Replace placeholders
    plist_content = template.format(
        WORKING_DIR=str(working_dir),
        VENV_PATH=str(venv_path),
        HOST=host,
        PORT=str(port),
        LOG_LEVEL=log_level,
        LOG_DIR=log_dir,
    )

    # Install to LaunchAgents (user-level)
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / "com.heylookllm.server.plist"

    # Write plist file
    try:
        plist_path.write_text(plist_content)
        logger.info(f"Service file written to: {plist_path}")
    except PermissionError:
        logger.error(f"Permission denied writing to {plist_path}")
        return False

    # Load the service
    try:
        # Unload first if it exists (ignore errors)
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True,
        )
        # Load the service
        subprocess.run(
            ["launchctl", "load", str(plist_path)],
            check=True,
        )
        logger.info("Service loaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to load service: {e}")
        return False


def uninstall_service_linux(system_wide: bool = False) -> bool:
    """Uninstall systemd service on Linux."""
    try:
        if system_wide:
            subprocess.run(["systemctl", "stop", "heylookllm.service"], check=False)
            subprocess.run(["systemctl", "disable", "heylookllm.service"], check=False)
            service_path = Path("/etc/systemd/system/heylookllm.service")
        else:
            subprocess.run(["systemctl", "--user", "stop", "heylookllm.service"], check=False)
            subprocess.run(["systemctl", "--user", "disable", "heylookllm.service"], check=False)
            service_path = Path.home() / ".config" / "systemd" / "user" / "heylookllm.service"

        if service_path.exists():
            service_path.unlink()
            logger.info(f"Removed service file: {service_path}")

        if system_wide:
            subprocess.run(["systemctl", "daemon-reload"], check=True)
        else:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)

        logger.info("Service uninstalled successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to uninstall service: {e}")
        return False


def uninstall_service_macos() -> bool:
    """Uninstall launchd service on macOS."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.heylookllm.server.plist"

    try:
        if plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True,
            )
            plist_path.unlink()
            logger.info(f"Removed service file: {plist_path}")

        logger.info("Service uninstalled successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to uninstall service: {e}")
        return False


def start_service_linux(system_wide: bool = False) -> bool:
    """Start the systemd service."""
    try:
        if system_wide:
            subprocess.run(["systemctl", "start", "heylookllm.service"], check=True)
        else:
            subprocess.run(["systemctl", "--user", "start", "heylookllm.service"], check=True)
        logger.info("Service started")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start service: {e}")
        return False


def start_service_macos() -> bool:
    """Start the launchd service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.heylookllm.server.plist"
    try:
        subprocess.run(["launchctl", "start", "com.heylookllm.server"], check=True)
        logger.info("Service started")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start service: {e}")
        return False


def stop_service_linux(system_wide: bool = False) -> bool:
    """Stop the systemd service."""
    try:
        if system_wide:
            subprocess.run(["systemctl", "stop", "heylookllm.service"], check=True)
        else:
            subprocess.run(["systemctl", "--user", "stop", "heylookllm.service"], check=True)
        logger.info("Service stopped")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop service: {e}")
        return False


def stop_service_macos() -> bool:
    """Stop the launchd service."""
    try:
        subprocess.run(["launchctl", "stop", "com.heylookllm.server"], check=True)
        logger.info("Service stopped")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop service: {e}")
        return False


def status_service_linux(system_wide: bool = False) -> dict:
    """Get status of the systemd service."""
    try:
        if system_wide:
            result = subprocess.run(
                ["systemctl", "status", "heylookllm.service"],
                capture_output=True,
                text=True,
            )
            is_active = subprocess.run(
                ["systemctl", "is-active", "heylookllm.service"],
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                ["systemctl", "--user", "status", "heylookllm.service"],
                capture_output=True,
                text=True,
            )
            is_active = subprocess.run(
                ["systemctl", "--user", "is-active", "heylookllm.service"],
                capture_output=True,
                text=True,
            )

        return {
            "installed": "could not be found" not in result.stderr.lower(),
            "active": is_active.stdout.strip() == "active",
            "status_output": result.stdout,
        }
    except Exception as e:
        return {"installed": False, "active": False, "error": str(e)}


def status_service_macos() -> dict:
    """Get status of the launchd service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.heylookllm.server.plist"

    try:
        result = subprocess.run(
            ["launchctl", "list", "com.heylookllm.server"],
            capture_output=True,
            text=True,
        )

        installed = plist_path.exists()
        # If launchctl list succeeds and shows PID, it's running
        active = result.returncode == 0 and "PID" not in result.stderr

        # Parse the output for more details
        status_lines = result.stdout.strip().split("\n") if result.stdout else []

        return {
            "installed": installed,
            "active": active and installed,
            "status_output": result.stdout if result.stdout else result.stderr,
        }
    except Exception as e:
        return {"installed": False, "active": False, "error": str(e)}


def print_status(status: dict):
    """Print service status in a readable format."""
    if status.get("error"):
        print(f"Error checking status: {status['error']}")
        return

    print(f"Installed: {'Yes' if status['installed'] else 'No'}")
    print(f"Running: {'Yes' if status['active'] else 'No'}")

    if status.get("status_output"):
        print("\nDetails:")
        print(status["status_output"])


def manage_service(args) -> int:
    """Main entry point for service management CLI.

    Returns exit code (0 for success, 1 for failure).
    """
    platform = get_platform()

    if platform == "unsupported":
        print(f"Error: Platform '{sys.platform}' is not supported for service management.")
        print("Supported platforms: macOS, Linux (Ubuntu/Debian/etc.)")
        return 1

    action = args.action

    if action == "install":
        # Security warning if binding to all interfaces
        if args.host == "0.0.0.0":
            print("WARNING: Binding to 0.0.0.0 exposes the server to your network.")
            print("Ensure you have firewall rules in place and are behind a VPN.")
            print("See: guides/SERVICE_SECURITY.md for configuration guidance.")
            print()

        if platform == "linux":
            success = install_service_linux(
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                log_dir=args.log_dir,
                system_wide=getattr(args, 'system_wide', False),
            )
        else:  # macos
            success = install_service_macos(
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                log_dir=args.log_dir,
            )

        if success:
            print("Service installed successfully!")
            print()
            if platform == "linux":
                print("Start the service with:")
                if getattr(args, 'system_wide', False):
                    print("  sudo systemctl start heylookllm")
                else:
                    print("  systemctl --user start heylookllm")
                print()
                print("View logs with:")
                if getattr(args, 'system_wide', False):
                    print("  journalctl -u heylookllm -f")
                else:
                    print("  journalctl --user -u heylookllm -f")
            else:
                print("The service will start automatically.")
                print()
                print("View logs in:", args.log_dir or str(Path.cwd() / "logs"))
            return 0
        return 1

    elif action == "uninstall":
        if platform == "linux":
            success = uninstall_service_linux(system_wide=getattr(args, 'system_wide', False))
        else:
            success = uninstall_service_macos()
        return 0 if success else 1

    elif action == "start":
        if platform == "linux":
            success = start_service_linux(system_wide=getattr(args, 'system_wide', False))
        else:
            success = start_service_macos()
        return 0 if success else 1

    elif action == "stop":
        if platform == "linux":
            success = stop_service_linux(system_wide=getattr(args, 'system_wide', False))
        else:
            success = stop_service_macos()
        return 0 if success else 1

    elif action == "restart":
        if platform == "linux":
            stop_service_linux(system_wide=getattr(args, 'system_wide', False))
            success = start_service_linux(system_wide=getattr(args, 'system_wide', False))
        else:
            stop_service_macos()
            success = start_service_macos()
        return 0 if success else 1

    elif action == "status":
        if platform == "linux":
            status = status_service_linux(system_wide=getattr(args, 'system_wide', False))
        else:
            status = status_service_macos()
        print_status(status)
        return 0

    else:
        print(f"Unknown action: {action}")
        return 1
