"""Contract: the DNS-rebinding guard (host_check.py) on the real app.

Claims (what breaks if a test is deleted):
- a request naming a host this server is not is refused before any route
  runs, with the fix named; the static frontend is covered too, since a
  rebound page loads it first;
- an IP address, localhost, the machine's own name and a configured name
  all pass -- the guard must never lock out a LAN client that uses an address;
- heylook.toml's ``allowed_hosts`` is read live through the router's config.
"""

import pytest

from heylook_llm.host_check import host_only, own_names


@pytest.mark.parametrize("path", ["/v1/models", "/"])
def test_a_foreign_host_is_refused(client, path):
    resp = client.get(path, headers={"Host": "rebind.attacker.example:8000"})
    assert resp.status_code == 403
    assert "allowed_hosts" in resp.json()["detail"]


@pytest.mark.parametrize("host", [
    "192.0.2.10:1263", "[::1]:8000", "127.0.0.1", "localhost:8000",
    f"{sorted(own_names() - {'localhost'})[0]}:8000", "testserver",
])
def test_addresses_and_known_names_pass(client, host):
    assert client.get("/v1/models", headers={"Host": host}).status_code == 200


def test_the_configured_list_is_read_live(client, mock_router):
    name = "mac.tailnet.example"
    assert client.get("/v1/models", headers={"Host": name}).status_code == 403
    mock_router.app_config.allowed_hosts.append(name.upper() + ".")
    try:
        assert client.get("/v1/models", headers={"Host": name}).status_code == 200
    finally:
        mock_router.app_config.allowed_hosts.remove(name.upper() + ".")


def test_host_only_strips_port_brackets_and_dot():
    assert host_only("Example.COM.:80") == "example.com"
    assert host_only("[fe80::1]:8000") == "fe80::1"
    assert host_only("::1") == "::1"
