"""Keep regression tests off the network and off live Braket submits."""
from __future__ import annotations

import socket

import pytest


def _refuse(*_args, **_kwargs):
    raise AssertionError("regression test attempted a network connection")


@pytest.fixture(autouse=True)
def forbid_network_and_braket_submit(monkeypatch):
    monkeypatch.delenv("BRAKET_SUBMIT", raising=False)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
