#!/usr/bin/env python3
"""
Network diagnostic script to help figure out connectivity between machines.
Run this on both the server machine and client machine.

Usage: python network_diagnostic.py
"""

import socket
import subprocess
import sys
import platform

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Connect to a remote address (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "Unable to determine"

def get_hostname():
    """Get the hostname of this machine."""
    try:
        return socket.gethostname()
    except Exception:
        return "Unable to determine"

def check_port_open(host, port):
    """Check if a port is open on a given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def ping_host(host):
    """Ping a host to check basic connectivity."""
    try:
        if platform.system().lower() == "windows":
            cmd = ["ping", "-n", "1", host]
        else:
            cmd = ["ping", "-c", "1", host]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def main():
    """Run network diagnostics."""
    print("🌐 Network Diagnostic for Edge-LLM")
    print("=" * 50)
    print()

    # Basic system info
    print("📋 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Hostname: {get_hostname()}")
    print(f"   Local IP: {get_local_ip()}")
    print()

    # Check if we're likely the server or client
    print("🔍 Checking for Edge-LLM server...")
    if check_port_open("localhost", 8080):
        print("✅ Port 8080 is open on localhost - this machine is running the server")
        server_mode = True
    else:
        print("❌ Port 8080 is not open on localhost - this machine is not running the server")
        server_mode = False
    print()

    if server_mode:
        print("🖥️  SERVER MODE - Edge-LLM is running here")
        print("   Other machines can connect to this server using:")
        local_ip = get_local_ip()
        print(f"   http://{local_ip}:8080")
        print()
        print("   To test from the client machine:")
        print(f"   python test_integration.py --server {local_ip}:8080")
        print()
        print("   Make sure your firewall allows incoming connections on port 8080")

    else:
        print("💻 CLIENT MODE - Looking for Edge-LLM server")
        print()

        # Test common server addresses
        test_addresses = [
            "localhost",
            "127.0.0.1",
            "10.0.0.100"
        ]

        print("🔍 Scanning for servers on port 8080...")
        found_servers = []

        for addr in test_addresses:
            print(f"   Testing {addr}:8080... ", end="", flush=True)
            if check_port_open(addr, 8080):
                print("✅ FOUND")
                found_servers.append(addr)
            else:
                print("❌")

        print()

        if found_servers:
            print("🎉 Found Edge-LLM server(s):")
            for server in found_servers:
                print(f"   http://{server}:8080")
                print(f"   Test with: python test_integration.py --server {server}:8080")
        else:
            print("❌ No Edge-LLM servers found on common addresses")
            print()
            print("💡 To find your server:")
            print("   1. Run this script on the server machine to get its IP")
            print("   2. Make sure the server is started with: heylookllm --host 0.0.0.0")
            print("   3. Check firewall settings on both machines")
            print("   4. Use: python test_integration.py --server SERVER_IP:8080")

    print()
    print("🛠️  Troubleshooting Tips:")
    print("   • Firewall: Make sure port 8080 is allowed")
    print("   • Server binding: Use --host 0.0.0.0 (not 127.0.0.1)")
    print("   • Network: Both machines should be on the same network")
    print("   • Testing: Try curl http://SERVER_IP:8080/v1/models")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
