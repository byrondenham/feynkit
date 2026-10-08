"""Tests for the Milnor fibres and Lê numbers of feynkit.milnor.

Oracles: the Milnor numbers of the simple singularities A_k, D_4 and E_6
(k, 4 and 6, by hand from the Jacobian ideals), a submersion, and the worked
examples of D. B. Massey, "Non-isolated hypersurface singularities and Lê
cycles", arXiv:1410.3312: Ex. 3.12 (p. 21) and Ex. 3.13 (pp. 21-22), whose
Lê numbers are stated there. Their Euler characteristics are not printed;
Ex. 4.2 (p. 23) asks for them, and the formula
chi(F) = 1 + sum_k (-1)^(n-k) lambda^k on p. 23 gives 2 and 5.

Hand examples: x y in four variables, a family of A_1 points over a smooth
plane (lambda = (0, 0, 1), chi(F) = chi(C^*) = 0), and Ex. 3.13 without w,
worked as Massey works Ex. 3.13 (lambda = (5, 4, 1), chi(F) = -1), whose
critical locus has dimension 2. In Massey's own coordinates for Ex. 3.12 the
polar number is a - b, so his Ex. 4.6(2) (p. 24) gives the Milnor number
b + (j - 1)(b - 1) of f + t^j for j > 1 + b/(a - b).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from fractions import Fraction

import pytest
import sympy as sp

from feynkit import milnor as milnor_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.landau import _singular_binary
from feynkit.milnor import LeNumbers, le_numbers, milnor_fibre_euler_characteristic

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

T, U, V, W, X, Y, Z = sp.symbols("t u v w x y z")


def origin(variables: list[sp.Symbol]) -> list[int]:
    return [0] * len(variables)


def massey_3_12(a: int, b: int) -> sp.Expr:
    """Massey's Ex. 3.12 (p. 21), y^2 - x^a - t x^b with a > b > 1, in (t, x, y)."""
    return Y**2 - X**a - T * X**b


# Massey's Ex. 3.13 (p. 21), in (u, v, w, x, y).
MASSEY_3_13 = Y**2 - X**3 - (U**2 + V**2 + W**2) * X**2

# Germs at the origin with a non-isolated critical point: (f, variables, Lê numbers, chi~(F)).
NON_ISOLATED = {
    "3.12, a = 3, b = 2": (massey_3_12(3, 2), [T, X, Y], (2, 1), 1),
    "3.12, a = 5, b = 3": (massey_3_12(5, 3), [T, X, Y], (3, 2), 1),
    "3.13": (MASSEY_3_13, [U, V, W, X, Y], (5, 4, 4, 1), 4),
    "3.13 without w": (Y**2 - X**3 - (U**2 + V**2) * X**2, [U, V, X, Y], (5, 4, 1), -2),
    "x y in four variables": (X * Y, [X, Y, Z, W], (0, 0, 1), -1),
}


# --- isolated critical points and submersions -------------------------------------------------


@requires_singular
class TestIsolated:
    """chi~(F) = (-1)^n mu at an isolated critical point, n + 1 the number of variables."""

    @pytest.mark.parametrize(
        ("name", "f", "variables", "mu"),
        [
            ("A_1", X**2 + Y**2, [X, Y], 1),
            ("A_3", X**4 + Y**2, [X, Y], 3),
            ("A_4", X**5 + Y**2, [X, Y], 4),
            ("D_4", X**2 * Y + Y**3, [X, Y], 4),
            ("E_6", X**3 + Y**4, [X, Y], 6),
            ("A_2 in three variables", X**3 + Y**2 + Z**2, [X, Y, Z], 2),
        ],
    )
    def test_milnor_numbers(
        self, name: str, f: sp.Expr, variables: list[sp.Symbol], mu: int
    ) -> None:
        le = le_numbers(f, variables, origin(variables))
        n = len(variables) - 1
        assert le.method == "isolated"
        assert le.critical_dimension == 0
        assert le.numbers == (mu,)
        assert le.polar_numbers == ()
        assert le.reduced_euler_characteristic == (-1) ** n * mu
        assert milnor_fibre_euler_characteristic(f, variables, origin(variables)) == (-1) ** n * mu

    def test_isolated_coordinates_are_the_variables(self) -> None:
        le = le_numbers(X**3 + Y**4, [X, Y], [0, 0])
        assert le.flag == ((1, 0), (0, 1))
        assert le.prime is not None and le.prime < 2**29

    def test_a_rational_point(self) -> None:
        half = sp.Rational(1, 2)
        f = (X - half) ** 3 + (Y + 2) ** 4
        assert milnor_fibre_euler_characteristic(f, [X, Y], [half, -2]) == -6
        assert le_numbers(f, [X, Y], [Fraction(1, 2), -2]).numbers == (6,)


class TestSubmersion:
    def test_a_submersion_has_contractible_fibre(self) -> None:
        f = X + Y**2
        assert milnor_fibre_euler_characteristic(f, [X, Y], [0, 0]) == 0
        le = le_numbers(f, [X, Y], [0, 0])
        assert le == LeNumbers((), (), -1, ((1, 0), (0, 1)), None, "submersion")
        assert le.reduced_euler_characteristic == 0

    def test_a_smooth_point_of_a_singular_hypersurface(self) -> None:
        # x y vanishes at (1, 0), where its gradient is (0, 1).
        assert milnor_fibre_euler_characteristic(X * Y, [X, Y], [1, 0]) == 0


# --- Massey's examples ------------------------------------------------------------------------


@requires_singular
class TestMasseyExamples:
    @pytest.mark.parametrize(("a", "b"), [(3, 2), (4, 2), (5, 3)])
    def test_example_3_12(self, a: int, b: int) -> None:
        # p. 21: lambda^0 = b, lambda^1 = b - 1; n = 2, so chi(F) = 1 + b - (b - 1) = 2.
        f = massey_3_12(a, b)
        le = le_numbers(f, [T, X, Y], [0, 0, 0])
        assert le.method == "attaching"
        assert le.critical_dimension == 1
        assert le.numbers == (b, b - 1)
        assert le.reduced_euler_characteristic == 2 - 1
        assert milnor_fibre_euler_characteristic(f, [T, X, Y], [0, 0, 0]) == 2 - 1

    def test_example_3_13(self) -> None:
        # pp. 21-22: lambda^3 = 1, lambda^2 = 4, lambda^1 = 4, lambda^0 = 5; n = 4, so
        # chi(F) = 1 + 5 - 4 + 4 - 1 = 5.
        variables = [U, V, W, X, Y]
        le = le_numbers(MASSEY_3_13, variables, origin(variables))
        assert le.critical_dimension == 3
        assert le.numbers == (5, 4, 4, 1)
        assert le.reduced_euler_characteristic == 5 - 1
        assert milnor_fibre_euler_characteristic(MASSEY_3_13, variables, origin(variables)) == 4


# --- agreement of the routes ------------------------------------------------------------------


Run = Callable[[str, str, float], tuple[str, bool]]


def spy(
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[str, str, bool], tuple[str, bool]] | None = None,
) -> list[str]:
    """Record the scripts Singular runs; ``change(script, output, timed_out)`` may alter what
    a run returns."""
    scripts: list[str] = []
    real: Run = milnor_module._run

    def run(script: str, binary: str, timeout: float) -> tuple[str, bool]:
        scripts.append(script)
        text, timed_out = real(script, binary, timeout)
        return (text, timed_out) if change is None else change(script, text, timed_out)

    monkeypatch.setattr(milnor_module, "_run", run)
    return scripts


def is_check(script: str) -> bool:
    return 'print("J' in script


def is_first_level_check(script: str) -> bool:
    """The Iomdine-Lê check at level 0 when s = 2, a polar computation."""
    return is_check(script) and "G = satq(P" in script


@requires_singular
class TestAgreement:
    @pytest.mark.parametrize("name", NON_ISOLATED)
    def test_attaching_sum_le_numbers_and_iomdine_le_agree(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f, variables, numbers, chi = NON_ISOLATED[name]
        scripts = spy(monkeypatch)
        le = le_numbers(f, variables, origin(variables))
        assert sum(map(is_check, scripts)) == (2 if len(numbers) == 3 else 1)
        assert le.method == "attaching"
        assert le.numbers == numbers
        assert le.critical_dimension == len(numbers) - 1
        # Thm 4.1 from the Lê numbers.
        assert le.reduced_euler_characteristic == chi
        scripts.clear()
        # The attaching sum, checked against Thm 4.1 at two flags, and the Iomdine-Lê check.
        assert milnor_fibre_euler_characteristic(f, variables, origin(variables)) == chi
        checks = [script for script in scripts if is_check(script)]
        assert len(checks) == (2 if le.critical_dimension == 2 else 1)
        assert sum(map(is_first_level_check, checks)) == (le.critical_dimension == 2)

    @pytest.mark.parametrize("name", NON_ISOLATED)
    def test_two_seeds_agree(self, name: str) -> None:
        f, variables, _, _ = NON_ISOLATED[name]
        first, second = (le_numbers(f, variables, origin(variables), seed=s) for s in (0, 1))
        assert first.flag != second.flag
        assert (first.numbers, first.polar_numbers) == (second.numbers, second.polar_numbers)
        assert le_numbers(f, variables, origin(variables), seed=0) == first

    def test_the_two_flags_differ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both functions compute two flags with different random coordinates, at two primes."""
        draws: dict[int, milnor_module._Draw] = {}
        real = milnor_module._draw

        def draw(seed: int, flag: int, attempt: int, m: int) -> milnor_module._Draw:
            draws[flag] = real(seed, flag, attempt, m)
            return draws[flag]

        monkeypatch.setattr(milnor_module, "_draw", draw)
        scripts = spy(monkeypatch)
        f, variables, _, _ = NON_ISOLATED["3.13"]
        for compute in (le_numbers, milnor_fibre_euler_characteristic):
            draws.clear()
            scripts.clear()
            compute(f, variables, origin(variables))
            assert set(draws) == {0, 1}
            assert draws[0].shears != draws[1].shears
            assert draws[0].hyperplanes != draws[1].hyperplanes
            flags = [script for script in scripts if not is_check(script)]
            primes = {p for script in flags for p in re.findall(r"ring S = (\d+),", script)}
            assert len(flags) == 2 and len(primes) == 2

    def test_the_flag(self) -> None:
        le = le_numbers(MASSEY_3_13, [U, V, W, X, Y], [0] * 5)
        m = 5
        for k, row in enumerate(le.flag):
            assert row[:k] == (0,) * k and row[k] == 1
            if k < le.critical_dimension:
                assert all(0 < c < 2**28 for c in row[k + 1 :])
            else:
                assert row[k + 1 :] == (0,) * (m - 1 - k)

    @pytest.mark.parametrize(("a", "b"), [(3, 2), (4, 2), (5, 3)])
    def test_iomdine_le_in_massey_coordinates(self, a: int, b: int) -> None:
        # In (t, x, y) gamma^1 = (V(a x^(a-b) + b t, y) . V(t))_0 = a - b and lambda = (b, b - 1).
        first = b // (a - b) + 2
        for j in (first, first + 1):
            f = massey_3_12(a, b) + T**j
            mu = b + (j - 1) * (b - 1)
            assert le_numbers(f, [T, X, Y], [0, 0, 0]).numbers == (mu,)
            assert milnor_fibre_euler_characteristic(f, [T, X, Y], [0, 0, 0]) == mu


# --- undecided and failing runs ---------------------------------------------------------------


@requires_singular
class TestUndecided:
    """Coordinates that fail Def. 3.8's conditions, checks and runs past the limit."""

    # In (x, t, y), z_0 = x contains the critical line, the t-axis, so restricting to V(x)
    # leaves y^2, whose critical locus is still a line.
    F = massey_3_12(3, 2)
    VARIABLES = [X, T, Y]

    @staticmethod
    def variables_as_coordinates(monkeypatch: pytest.MonkeyPatch, attempts: set[int]) -> None:
        """Make the coordinates of these attempts of every flag the variables themselves."""
        real = milnor_module._draw

        def draw(seed: int, flag: int, attempt: int, m: int) -> milnor_module._Draw:
            found = real(seed, flag, attempt, m)
            if attempt not in attempts:
                return found
            return dataclasses.replace(found, shears=tuple((0,) * len(r) for r in found.shears))

        monkeypatch.setattr(milnor_module, "_draw", draw)

    def test_a_flag_failing_def_3_8_is_redrawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.variables_as_coordinates(monkeypatch, {0})
        le = le_numbers(self.F, self.VARIABLES, [0, 0, 0])
        assert le.numbers == (2, 1)
        assert le.flag[0] != (1, 0, 0)
        assert milnor_fibre_euler_characteristic(self.F, self.VARIABLES, [0, 0, 0]) == 1

    def test_two_failing_flags_are_undecided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.variables_as_coordinates(monkeypatch, {0, 1})
        message = (
            "undecided: 2 flags .* level 1: the critical locus of the restriction has "
            "dimension 1, not 0"
        )
        with pytest.raises(ComputationError, match=message):
            le_numbers(self.F, self.VARIABLES, [0, 0, 0])
        with pytest.raises(ComputationError, match=message):
            milnor_fibre_euler_characteristic(self.F, self.VARIABLES, [0, 0, 0])

    def test_a_run_past_the_limit_is_undecided(self) -> None:
        with pytest.raises(ComputationError, match=r"undecided: Singular ran past timeout=1e-06 s"):
            le_numbers(self.F, self.VARIABLES, [0, 0, 0], timeout=1e-6)

    def test_the_level_of_a_run_past_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            milnor_module, "_run", lambda *args: ("D 0 1\nP 0 1 3 1 2\nD 1 0\n", True)
        )
        with pytest.raises(ComputationError, match="timeout=120 s at level 1 of the flag"):
            milnor_fibre_euler_characteristic(self.F, self.VARIABLES, [0, 0, 0])

    def test_a_failing_iomdine_le_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def wrong(script: str, text: str, timed_out: bool) -> tuple[str, bool]:
            if is_check(script):
                text = re.sub(r"J (\d+) (\d+)", lambda m: f"J {m[1]} {int(m[2]) + 1}", text)
            return text, timed_out

        spy(monkeypatch, wrong)
        message = "undecided: the Iomdine-Lê check fails at level 0, j = 4"
        with pytest.raises(ComputationError, match=message):
            milnor_fibre_euler_characteristic(self.F, self.VARIABLES, [0, 0, 0])
        with pytest.raises(ComputationError, match=message):
            le_numbers(self.F, self.VARIABLES, [0, 0, 0])
        assert (
            milnor_fibre_euler_characteristic(self.F, self.VARIABLES, [0, 0, 0], check=False) == 1
        )

    def test_the_last_level_check_must_finish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy(
            monkeypatch,
            lambda script, text, timed_out: ("", True) if is_check(script) else (text, timed_out),
        )
        with pytest.raises(
            ComputationError, match="undecided: the Iomdine-Lê check at level 0 ran past"
        ):
            milnor_fibre_euler_characteristic(self.F, self.VARIABLES, [0, 0, 0])

    def test_the_first_level_check_may_pass_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f, variables, _, chi = NON_ISOLATED["3.13 without w"]
        scripts = spy(
            monkeypatch,
            lambda script, text, timed_out: (
                ("", True) if is_first_level_check(script) else (text, timed_out)
            ),
        )
        assert milnor_fibre_euler_characteristic(f, variables, origin(variables)) == chi
        assert sum(map(is_first_level_check, scripts)) == 1

    def test_flags_that_disagree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = milnor_module._flag

        def flag(
            germ: milnor_module._Germ, p: int, seed: int, index: int, binary: str, timeout: float
        ) -> milnor_module._Flag:
            found = real(germ, p, seed, index, binary, timeout)
            if index == 0:
                return found
            wrong = tuple(value + 1 for value in found.le.numbers)
            return dataclasses.replace(found, le=dataclasses.replace(found.le, numbers=wrong))

        monkeypatch.setattr(milnor_module, "_flag", flag)
        # The second flag keeps its attaching sum, so the Euler characteristics agree; the Lê
        # numbers do not, and generic ones would, so neither function picks one.
        message = r"^undecided: two flags give the Lê numbers \(2, 1\) and \(3, 2\)"
        with pytest.raises(ComputationError, match=message):
            milnor_fibre_euler_characteristic(self.F, self.VARIABLES, [0, 0, 0])
        with pytest.raises(ComputationError, match=message):
            le_numbers(self.F, self.VARIABLES, [0, 0, 0])

    @pytest.mark.parametrize(
        ("output", "reason"),
        [
            ("", "no critical dimension"),
            ("D 0 3", "dimension 3 at the point"),
            ("D 0 1", "level 0: no polar curve"),
            ("D 0 1\nP 0 2 -1 -1 -1", "dimension 2 or more"),
            ("D 0 1\nP 0 1 -1 1 2", "in a curve"),
            ("D 0 1\nP 0 1 4 1 2", "Teissier split fails, 4 != 1 + 2"),
            ("D 0 1\nP 0 1 3 1 2", "level 1: no critical dimension"),
            ("D 0 1\nP 0 1 3 1 2\nD 1 0\nI 1 -1", "level 1: no finite Milnor number"),
        ],
    )
    def test_conditions_on_the_numbers(self, output: str, reason: str) -> None:
        # n = 2, three variables.
        result = milnor_module._judge(milnor_module._read(output), 2)
        assert isinstance(result, str) and reason in result

    def test_a_negative_le_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # lambda^1 = mu - gamma^1 = 0 - 1.
        monkeypatch.setattr(
            milnor_module, "_run", lambda *args: ("D 0 1\nP 0 1 3 1 2\nD 1 0\nI 1 0\n", False)
        )
        with pytest.raises(
            ComputationError, match=r"^undecided: 2 flags .*a negative Lê number \(2, -1\)"
        ):
            le_numbers(self.F, self.VARIABLES, [0, 0, 0])

    def test_a_singular_error_raises(self) -> None:
        binary = _singular_binary()
        assert binary is not None
        with pytest.raises(ComputationError, match="^undecided: Singular failed"):
            milnor_module._run("ring r = 0, (x), dp;\npoly f = x +;\nquit;\n", binary, 10)


class TestWithoutSingular:
    @pytest.fixture(autouse=True)
    def no_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(milnor_module, "_singular_binary", lambda: None)

    def test_a_critical_point_needs_singular(self) -> None:
        with pytest.raises(RuntimeError, match="Singular"):
            le_numbers(X**3 + Y**4, [X, Y], [0, 0])
        with pytest.raises(RuntimeError, match="Singular"):
            milnor_fibre_euler_characteristic(X**3 + Y**4, [X, Y], [0, 0])

    def test_a_submersion_does_not(self) -> None:
        assert milnor_fibre_euler_characteristic(X + Y**2, [X, Y], [0, 0]) == 0


# --- input ------------------------------------------------------------------------------------


class TestInput:
    @pytest.mark.parametrize(
        ("f", "variables", "at", "message"),
        [
            (X**2, [], [], "variables"),
            (X**2, [X, X], [0, 0], "variables"),
            (X**2, [X, 2], [0, 0], "variables"),
            (X**2 + Y, [X], [0], "polynomial"),
            (1 / X, [X], [0], "polynomial"),
            (sp.sqrt(2) * X**2, [X], [0], "polynomial"),
            ("x**2 +", [X], [0], "polynomial"),
            (sp.Integer(0), [X], [0], "zero"),
            (X**2, [X], [0, 0], "one rational number"),
            (X**2, [X], 0, "one rational number"),
            (X**2, [X], "0", "one rational number"),
            (X**2, [X], [0.5], "rational"),
            (X**2, [X], [True], "rational"),
            (X**2, [X], [sp.sqrt(2)], "rational"),
            (X**2 - 1, [X], [0], "vanish"),
        ],
    )
    def test_bad_input(self, f: object, variables: list, at: object, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            milnor_fibre_euler_characteristic(f, variables, at)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match=message):
            le_numbers(f, variables, at)  # type: ignore[arg-type]

    @pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), "5", 3e6, None])
    def test_timeout_must_be_a_number_of_seconds(self, timeout: object) -> None:
        with pytest.raises(ValidationError, match="timeout"):
            le_numbers(X**2, [X], [0], timeout=timeout)  # type: ignore[arg-type]

    @pytest.mark.parametrize("seed", [1.5, "0", True, None])
    def test_seed_must_be_an_integer(self, seed: object) -> None:
        with pytest.raises(ValidationError, match="seed"):
            milnor_fibre_euler_characteristic(X**2, [X], [0], seed=seed)  # type: ignore[arg-type]
