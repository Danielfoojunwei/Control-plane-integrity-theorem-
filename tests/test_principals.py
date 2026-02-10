"""Tests for principal trust classification."""

import pytest

from src.principals import (
    Principal,
    is_trusted_principal,
    _TRUSTED_PRINCIPALS,
    UNTRUSTED_PRINCIPALS,
)


class TestPrincipalTrust:
    def test_sys_is_trusted(self):
        assert is_trusted_principal(Principal.SYS)

    def test_user_is_trusted(self):
        assert is_trusted_principal(Principal.USER)

    def test_web_is_untrusted(self):
        assert not is_trusted_principal(Principal.WEB)

    def test_skill_is_untrusted(self):
        assert not is_trusted_principal(Principal.SKILL)

    def test_tool_output_is_untrusted(self):
        assert not is_trusted_principal(Principal.TOOL_OUTPUT)

    def test_trusted_and_untrusted_are_disjoint(self):
        assert _TRUSTED_PRINCIPALS & UNTRUSTED_PRINCIPALS == frozenset()

    def test_all_principals_classified(self):
        all_principals = set(Principal)
        classified = _TRUSTED_PRINCIPALS | UNTRUSTED_PRINCIPALS
        assert all_principals == classified
