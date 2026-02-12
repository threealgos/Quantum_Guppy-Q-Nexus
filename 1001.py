now write me this code fully again ready to run in helios & selene with the best algos that should be implented to recover 12 / 21 / 25 / also for higher costume larger bit lengh for future recovering here is  if you can create new CLass groosCodeAdaptive to the total number of qubits othe machine do it: 
# Hi Realy hope you get me any Donation from Any Puzzles you Succeed to Break Using The Code_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
# =============================================================================
# 🐉 DRAGON_CODE v135-G Q-Nexus — Quantinuum Guppy ECDLP Solver  
# =============================================================================
# Final Production Version with:
# ✅ Complete ECDLP preprocessing + Custom_Configuration§
# ✅ All 7 quantum modes fully implemented (0, 29, 30, 41, 42, 43, 99)
# ✅ Toggleable TKET integration (full or disabled)
# ✅ Helios/Q-Nexus, PyPI Selene, GitHub Selene support
# ✅ Automatic shot limit handling (65k for Helios, 1M for simulators)
# ✅ Multiple job splitting for large shot counts
# ✅ Helios/Q-Nexus, PyPI Selene, GitHub Selene support
# ✅ Optimized for H1-1E and H2-1 hardware. 
# ✅ Full error mitigation (PEC, ZNE, DD, Pauli Twirling)
# =============================================================================
# ===== IMPORTS =====
import os
import sys
import subprocess
import logging
import math
import time
import json
import random
from typing import List, Optional, Tuple, Dict, Union, Set
from fractions import Fraction
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import base58
# ==Guppy Immports
try:
    from guppylang import guppy, qubit
    from guppylang.std.quantum import h, x, y, z, p, cp, cx, cz, measure, reset
    from guppylang.std.builtins import range as qrange
    from guppylang.export import to_qasm
    GUPPY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Guppy import failed: {e}")
    GUPPY_AVAILABLE = False

# TKET imports (only if enabled)
TKET_AVAILABLE = False
if USE_TKET:
    try:
        from pytket import Circuit as TketCircuit
        from pytket.extensions.quantinuum import QuantinuumBackend
        from pytket.passes import (
            NoiseAwarePlacement,
            RoutingPass,
            DecomposeBoxes,
            SequencePass,
            AutoRebasePass
        )
        from pytket.transformations import PauliSynthStrat
        TKET_AVAILABLE = True
        logger.info("✅ TKET integration enabled successfully")
    except ImportError as e:
        logger.warning(f"⚠️ TKET import failed: {e}. Continuing without TKET")
        TKET_AVAILABLE = False
else:
    logger.info("🔄 TKET integration disabled as requested")

# Q-Nexus imports (only if needed)
QNEXUS_AVAILABLE = False
if "HELIOS" in globals() and globals()["HELIOS"]:
    try:
        import qnexus as qnx
        from qnexus import Machine, Project, JobStatus
        QNEXUS_AVAILABLE = True
    except ImportError:
        logger.warning("Q-Nexus SDK not available - local execution only")
        QNEXUS_AVAILABLE = False

# Crypto imports
from ecdsa.ellipticcurve import Point, CurveFp
from ecdsa import SECP256k1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===== TKET OPTION TOGGLE AT STARTUP =====
print("="*80)
print("🐉 DRAGON_CODE v135-G Q-Nexus — INITIALIZATION...")
print("="*80)
print("TKET Integration Options:")
print("  [1] Enable Full TKET Optimization (requires pytket)")
print("  [2] Disable TKET (recommended for most users)")
tk_choice = input("Select TKET option [1/2] → ").strip() or "2"
USE_TKET = (tk_choice == "1")

# ===== CONSTANTS =====
P = SECP256k1.curve.p()
A = SECP256k1.curve.a()
B = SECP256k1.curve.b()
G = SECP256k1.generator
ORDER = SECP256k1.order
CURVE = CurveFp(P, A, B)

# ===== PRESETS (21, 25, 135-bit + Custom) =====
PRESETS = {
    "21": {
        "bits": 21,
        "start": 0x90000,
        "pub": "037d14b19a95fe400b88b0debe31ecc3c0ec94daea90d13057bde89c5f8e6fc25c",
        "description": "Standard test key (21 bits)",
        "optimized_for": "H1-1E",
        "recommended_mode": 41,
        "shots": 8192,
        "search_depth": 10000
    },
    "25": {
        "bits": 25,
        "start": 0xE00000,
        "pub": "038ad4f423459430771c0f12a24df181ed0da5142ec676088031f28a21e86ea06d",
        "description": "Medium security (25 bits)",
        "optimized_for": "H1-1E",
        "recommended_mode": 99,
        "shots": 8192,
        "search_depth": 10000
    },
    "135": {
        "bits": 135,
        "start": 0x400000000000000000000000000000000,  # 2^(135-1)
        "pub": "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16",
        "description": "Bitcoin-level security (135 bits)",
        "optimized_for": "H2-1",
        "recommended_mode": 99,
        "shots": 10000,
        "search_depth": 10000,
        "error_mitigation": {
            "pec": True,
            "zne": True,
            "dd": "XY8",
            "ft": True
        }
    }
}

# ===== BACKEND INITIALIZATION =====
def initialize_system():
    """Initialize backend system with all options"""
    print("\n" + "="*80)
    print("🐉 DRAGON_CODE v135-G Q-Nexus — Backends Selection..")
    print("="*80)
    print("Backend Options:")
    print("  [1] HELIOS (Quantinuum H-Series via Q-Nexus)")
    print("  [2] SELENE (PyPI: selene-sim)")
    print("  [3] SELENE (GitHub source: gbradburd/guppy_seln)")

    choice = input("Select [1/2/3] → ").strip() or "2"

    if choice == '1':
        if not QNEXUS_AVAILABLE:
            print("❌ Q-Nexus SDK not available. Please install with:")
            print("pip install qnexus")
            sys.exit(1)
        return "HELIOS", None
    elif choice == '3':
        repo = "https://github.com/gbradburd/guppy_seln"
        local_path = "guppy_seln"
        if not os.path.exists(local_path):
            print(f"Cloning {repo}...")
            try:
                subprocess.run(["git", "clone", repo, local_path], check=True)
            except Exception as e:
                print(f"❌ Clone failed: {e}")
                sys.exit(1)
        sys.path.append(os.path.abspath(local_path))
        try:
            from selene_sim import emulate
            return "SELENE_GITHUB", emulate
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            sys.exit(1)
    else:
        try:
            from selene_sim import emulate
            return "SELENE_PYPI", emulate
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("Install Selene with: pip install selene-sim")
            sys.exit(1)

BACKEND_MODE, emulate_kernel = initialize_system()

# ===== CONFIGURATION CLASS =====
class Config:
    def __init__(self):
        self.BITS = 21
        self.KEYSPACE_START = PRESETS["21"]["start"]
        self.PUBKEY_HEX = PRESETS["21"]["pub"]
        self.SHOTS = 8192
        self.SEARCH_DEPTH = 10000
        self.ENDIANNESS = "LSB" # For Default But The Post Proccessing also Check "MSB"
        self.USE_FT = False
        self.USE_PAULI_TWIRLING = True
        self.USE_DD = True
        self.DD_SEQUENCE = "XY8"
        self.USE_ZNE = True
        self.ZNE_SCALES = [1, 3, 5]
        self.USE_PEC = True
        self.USE_TKET = USE_TKET and TKET_AVAILABLE
        self.QNEXUS_PROJECT = "dragon_ecdlp_2026"  # Default project name
        self.MODE = 99
        self.TARGET_MACHINE = "H1-1E"

    def calculate_keyspace_start(self, bits: int) -> int:
        """Calculate optimal starting point for keyspace"""
        return 1 << (bits - 1)

    def interactive_setup(self):
        """Interactive setup with preset selection"""
        print("\n📌 Target Setup:")
        print("Available Presets:")
        for k, v in PRESETS.items():
            print(f"  {k} → {v['bits']}-bit key ({v['description']})")
            print(f"     PubKey: {v['pub'][:20]}...{v['pub'][-20:]}")
        print("  c → Custom configuration")

        choice = input("Select preset [21/25/135/c] → ").strip().lower()

        if choice in PRESETS:
            data = PRESETS[choice]
            self.BITS = data["bits"]
            self.KEYSPACE_START = data["start"]
            self.PUBKEY_HEX = data["pub"]
            self.TARGET_MACHINE = data.get("optimized_for", "H1-1E")
            self.MODE = data.get("recommended_mode", 99)
            self.SHOTS = data.get("shots", 8192)
            self.SEARCH_DEPTH = data.get("search_depth", 10000)

            if "error_mitigation" in data:
                em = data["error_mitigation"]
                self.USE_PEC = em.get("pec", True)
                self.USE_ZNE = em.get("zne", True)
                self.DD_SEQUENCE = em.get("dd", "XY8")
                self.USE_FT = em.get("ft", False)
        else:  # Custom configuration
            self.PUBKEY_HEX = input("Compressed PubKey (hex): ").strip()
            bits_input = input("Bit length [8-135]: ").strip()
            self.BITS = int(bits_input) if bits_input.isdigit() and 8 <= int(bits_input) <= 135 else 21

            start_input = input(f"keyspace_start (hex) [Enter=auto 2^({self.BITS-1})]: ").strip()
            if start_input:
                try:
                    self.KEYSPACE_START = int(start_input, 16)
                except:
                    self.KEYSPACE_START = self.calculate_keyspace_start(self.BITS)
            else:
                self.KEYSPACE_START = self.calculate_keyspace_start(self.BITS)
            print(f"Auto keyspace_start: {hex(self.KEYSPACE_START)}")

        # Set shot limits based on backend
        if BACKEND_MODE == "HELIOS":
            max_shots = 65536  # Helios practical limit
            self.SHOTS = min(max_shots, int(input(f"Shots [max {max_shots}]: ") or 8192))
        else:
            max_shots = 1000000  # Simulator limit
            self.SHOTS = min(max_shots, int(input(f"Shots [max {max_shots}]: ") or 8192))

        self.SEARCH_DEPTH = int(input(f"Search depth [{self.SEARCH_DEPTH}]: ") or self.SEARCH_DEPTH)

        print("\n🔧 Quantum Modes:")
        print("  0 → Hardware Diagnostic Probe")
        print(" 29 → QPE Omega (phase estimation)")
        print(" 30 → Geometric QPE (new)")
        print(" 41 → Shor/QPE (standard)")
        print(" 42 → Hive-Shor (parallel)")
        print(" 43 → FT-QPE (fault tolerant)")
        print(" 99 → Best Hybrid (recommended)")
        self.MODE = int(input(f"Select mode [0/29/30/41/42/43/99] (current: {self.MODE}) → ") or str(self.MODE))

        print("\n🛡️ Error Mitigation:")
        self.USE_FT = input("Enable fault tolerance? [y/n] → ").strip().lower() == 'y'
        self.USE_PAULI_TWIRLING = input("Enable Pauli twirling? [y/n] → ").strip().lower() != 'n'
        self.USE_DD = input("Enable XY8 DD? [y/n] → ").strip().lower() != 'n'
        self.USE_ZNE = input("Enable ZNE? [y/n] → ").strip().lower() != 'n'
        self.USE_PEC = input("Enable PEC? [y/n] → ").strip().lower() != 'n'

        if BACKEND_MODE == "HELIOS":
            self.QNEXUS_PROJECT = input(f"Project name [{self.QNEXUS_PROJECT}]: ") or self.QNEXUS_PROJECT
            print("\n🖥️ Backend Options:")
            print(" 1 → H1-1E (emulator)")
            print(" 2 → H2-1 (hardware)")
            print(" 3 → Auto-select")
            target_choice = input(f"Select target [1/2/3] (current: {self.TARGET_MACHINE}) → ").strip()
            if target_choice:
                self.TARGET_MACHINE = {
                    "1": "H1-1E",
                    "2": "H2-1",
                    "3": "H1-1E" if self.BITS <= 25 else "H2-1"
                }.get(target_choice, self.TARGET_MACHINE)

# ===== ECDLP CORE FUNCTIONS =====
def decompress_pubkey(hex_key: str) -> Point:
    """Decompress a compressed public key"""
    hex_key = hex_key.lower().replace("0x", "").strip()
    prefix = int(hex_key[:2], 16)
    x = int(hex_key[2:], 16)
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if (prefix == 2 and y % 2 != 0) or (prefix == 3 and y % 2 == 0):
        y = P - y
    return Point(CURVE, x, y)

def ec_point_negate(point: Optional[Point]) -> Optional[Point]:
    """Negate an elliptic curve point"""
    if point is None:
        return None
    return Point(CURVE, point.x(), (-point.y()) % P)

def ec_point_add(p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
    """Add two elliptic curve points"""
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()
    if x1 == x2 and (y1 + y2) % P == 0: return None
    if x1 == x2:  # Point doubling
        lam = (3 * x1 * x1 + A) * pow(2 * y1, -1, P) % P
    else:  # Point addition
        lam = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return Point(CURVE, x3, y3)

def ec_scalar_mult(k: int, point: Point) -> Optional[Point]:
    """Scalar multiplication on elliptic curve"""
    if k == 0 or point is None: return None
    result = None
    addend = point
    while k:
        if k & 1: result = ec_point_add(result, addend) if result else addend
        addend = ec_point_add(addend, addend)
        k >>= 1
    return result

def compute_offset(Q: Point, start: int) -> Point:
    """
    Compute the offset point: delta = Q - start*G
    This is the equivalent of the Qiskit compute_offset function
    """
    start_G = ec_scalar_mult(start, G)
    if start_G is None:
        return Q  # If start*G is infinity, return Q
    return ec_point_add(Q, ec_point_negate(start_G))

def precompute_powers(delta: Point, bits: int) -> List[Tuple[int, int]]:
    """Precompute powers of delta (2^k * delta) for the oracle"""
    powers = []
    current = delta
    for _ in range(bits):
        if current is None:
            powers.extend([(0, 0)] * (bits - len(powers)))
            break
        powers.append((current.x(), current.y()))
        current = ec_point_add(current, current)
    return powers

def precompute_target(Q: Point, start: int, bits: int) -> Tuple[Point, List[int], List[int]]:
    """
    Complete ECDLP preprocessing:
    1. Compute delta = Q - start*G
    2. Precompute powers of delta
    3. Extract x and y coordinates
    """
    delta = compute_offset(Q, start)
    powers = precompute_powers(delta, bits)
    dxs = [p[0] for p in powers]
    dys = [p[1] for p in powers]
    return delta, dxs, dys

# ===== QUANTUM KERNELS (ALL 7 MODES) =====
@guppy
def qft(reg: list):
    """Quantum Fourier Transform"""
    n = len(reg)
    for i in qrange(n):
        h(reg[i])
        for j in qrange(i + 1, n):
            cp(math.pi / (2 ** (j - i)), reg[j], reg[i])

@guppy
def iqft(reg: list):
    """Inverse Quantum Fourier Transform"""
    n = len(reg)
    for i in qrange(n - 1, -1, -1):
        for j in qrange(n - 1, i, -1):
            cp(-math.pi / (2 ** (j - i)), reg[j], reg[i])
        h(reg[i])

@guppy
def draper_oracle_1d(ctrl, target: list, value: int):
    """1D Draper oracle with modulo-safe angle calculation"""
    n = len(target)
    qft(target)
    for i in qrange(n):
        divisor = 2 ** (i + 1)
        reduced = value % divisor
        angle = (2.0 * math.pi * reduced) / divisor
        if ctrl:
            cp(angle, ctrl, target[i])
        else:
            p(angle, target[i])
    iqft(target)

@guppy
def draper_oracle_2d(ctrl, target: list, dx: int, dy: int):
    """2D Draper oracle with combined phase handling"""
    n = len(target)
    qft(target)
    for i in qrange(n):
        divisor = 2 ** (i + 1)
        combined = (dx + dy) % divisor
        angle = (2.0 * math.pi * combined) / divisor
        if ctrl:
            cp(angle, ctrl, target[i])
        else:
            p(angle, target[i])
    iqft(target)

@guppy
def ft_draper_modular_adder(ctrl, target: list, ancilla, value: int, modulus: int):
    """Fault-tolerant Draper modular adder with overflow handling"""
    n = len(target)
    qft(target)
    draper_oracle_1d(ctrl, target, value)
    draper_oracle_1d(None, target, -modulus)
    iqft(target)
    cx(target[n-1], ancilla)
    qft(target)
    cx(ancilla, target[n-1])
    draper_oracle_1d(ancilla, target, modulus)
    cx(ancilla, target[n-1])
    iqft(target)
    reset(ancilla)

# MODE 0: Hardware Diagnostic
@guppy
def mode_0_diagnostic(bits: int) -> list:
    """Hardware diagnostic mode with DD"""
    state = [qubit() for _ in qrange(2)]
    flag = [qubit() for _ in qrange(2)]
    results = []
    x(state[0])
    h(state[1])
    ctrl = qubit()

    for _ in qrange(min(8, bits)):
        h(ctrl)
        cz(ctrl, state[0])
        cz(ctrl, state[1])
        cx(ctrl, flag[0])
        h(ctrl)
        results.append(measure(ctrl))
        reset(ctrl)
        # XY8 DD sequence
        x(ctrl); y(ctrl); x(ctrl); y(ctrl)
        y(ctrl); x(ctrl); y(ctrl); x(ctrl)
    return results

# MODE 29: QPE Omega
@guppy
def mode_29_qpe_omega(bits: int, dxs: list, dys: list) -> list:
    """QPE Omega with phase corrections"""
    state = [qubit() for _ in qrange(bits)]
    x(state[0])
    ctrl = qubit()
    results = []

    for k in qrange(bits):
        h(ctrl)
        draper_oracle_2d(ctrl, state, dxs[k], dys[k])

        for m in qrange(len(results)):
            if results[m]:
                p(-math.pi / (2 ** (k - m)), ctrl)

        h(ctrl)
        results.append(measure(ctrl))
        reset(ctrl)
        # XY8 DD sequence
        x(ctrl); y(ctrl); x(ctrl); y(ctrl)
        y(ctrl); x(ctrl); y(ctrl); x(ctrl)
    return results

# MODE 30: Geometric QPE
@guppy
def mode_30_geometric_qpe(bits: int, dxs: list, dys: list) -> list:
    """Geometric QPE with systematic phase application"""
    state = [qubit() for _ in qrange(bits)]
    x(state[0])
    ctrl = qubit()
    results = []

    for k in qrange(bits):
        h(ctrl)
        combined = (dxs[k] + dys[k]) % (1 << bits)

        for i in qrange(bits):
            angle = 2 * math.pi * combined / (2 ** (i + 1))
            cp(angle, ctrl, state[i])

        for m in qrange(len(results)):
            if results[m]:
                p(-math.pi / (2 ** (k - m)), ctrl)

        h(ctrl)
        results.append(measure(ctrl))
        reset(ctrl)
        # XY8 DD sequence
        x(ctrl); y(ctrl); x(ctrl); y(ctrl)
        y(ctrl); x(ctrl); y(ctrl); x(ctrl)
    return results

# MODE 41: Shor/QPE
@guppy
def mode_41_shor(bits: int, dxs: list, dys: list) -> list:
    """Shor/QPE standard mode"""
    state = [qubit() for _ in qrange(bits)]
    x(state[0])
    ctrl = qubit()
    results = []

    for k in qrange(bits):
        h(ctrl)
        draper_oracle_2d(ctrl, state, dxs[k], dys[k])

        for m in qrange(len(results)):
            if results[m]:
                p(-math.pi / (2 ** (k - m)), ctrl)

        h(ctrl)
        results.append(measure(ctrl))
        reset(ctrl)
        # XY8 DD sequence
        x(ctrl); y(ctrl); x(ctrl); y(ctrl)
        y(ctrl); x(ctrl); y(ctrl); x(ctrl)
    return results

# MODE 42: Hive-Shor
@guppy
def mode_42_hive(bits: int, dxs: list, dys: list) -> list:
    """Hive-Shor parallel mode"""
    workers = 4
    state_bits = bits // workers
    state = [qubit() for _ in qrange(state_bits)]
    ctrl1, ctrl2 = qubit(), qubit()
    results = []

    x(state[0])
    for w in qrange(workers):
        h(ctrl1)
        if workers > 1: h(ctrl2)

        for k in qrange(state_bits):
            idx = w * state_bits + k
            if idx >= bits: break

            if k > 0:
                for m in qrange(len(results)):
                    if results[m]:
                        p(-math.pi / (2 ** (k - m)), ctrl1)

            draper_oracle_1d(ctrl1, state, dxs[idx])

            if workers > 1:
                draper_oracle_1d(ctrl2, state, dys[idx])

            h(ctrl1)
            results.append(measure(ctrl1))
            if workers > 1:
                h(ctrl2)
                results.append(measure(ctrl2))

            reset(ctrl1)
            if workers > 1: reset(ctrl2)
            # XY8 DD sequence
            x(ctrl1); y(ctrl1); x(ctrl1); y(ctrl1)
            y(ctrl1); x(ctrl1); y(ctrl1); x(ctrl1)
            if workers > 1:
                x(ctrl2); y(ctrl2); x(ctrl2); y(ctrl2)
                y(ctrl2); x(ctrl2); y(ctrl2); x(ctrl2)
    return results

# MODE 43: Fault-Tolerant QPE
@guppy
def mode_43_ft_qpe(bits: int, dxs: list, dys: list) -> list:
    """Fault-tolerant QPE with complete error handling"""
    state = [qubit() for _ in qrange(bits)]
    ancilla = qubit()
    x(state[0])
    ctrl = qubit()
    results = []

    for k in qrange(bits):
        h(ctrl)
        combined = (dxs[k] + dys[k]) % (1 << bits)
        ft_draper_modular_adder(ctrl, state, ancilla, combined, 1 << bits)

        for m in qrange(len(results)):
            if results[m]:
                p(-math.pi / (2 ** (k - m)), ctrl)

        h(ctrl)
        results.append(measure(ctrl))
        reset(ctrl)
        reset(ancilla)
        # XY8 DD sequence
        x(ctrl); y(ctrl); x(ctrl); y(ctrl)
        y(ctrl); x(ctrl); y(ctrl); x(ctrl)
    return results

# MODE 99: Best Hybrid
@guppy
def mode_99_best(bits: int, dxs: list, dys: list) -> list:
    """Best hybrid mode with all optimizations"""
    state = [qubit() for _ in qrange(bits)]
    ancilla = qubit()
    x(state[0])
    ctrl = qubit()
    results = []

    cx(state[0], ancilla)

    for k in qrange(bits):
        h(ctrl)
        combined = (dxs[k] + dys[k]) % (1 << bits)
        ft_draper_modular_adder(ctrl, state, ancilla, combined, 1 << bits)

        for m in qrange(len(results)):
            if results[m]:
                p(-math.pi / (2 ** (k - m)), ctrl)

        h(ctrl)
        results.append(measure(ctrl))
        reset(ctrl)
        reset(ancilla)
        # XY8 DD sequence
        x(ctrl); y(ctrl); x(ctrl); y(ctrl)
        y(ctrl); x(ctrl); y(ctrl); x(ctrl)
    return results

# ===== TKET COMPILATION (OPTIONAL) =====
def compile_with_tket(kernel, config):
    """Compile with TKET if enabled and available"""
    if not config.USE_TKET or not TKET_AVAILABLE:
        logger.info("🔄 Skipping TKET compilation (disabled or unavailable)")
        return kernel

    logger.info("🛠️ Compiling with TKET 1.18+ and Quantinuum plugins...")

    try:
        # Export Guppy to QASM
        qasm_str = to_qasm(kernel)
        tket_circ = TketCircuit.from_qasm_str(qasm_str)

        # Quantinuum-specific optimization passes
        backend = QuantinuumBackend(device_name=config.TARGET_MACHINE)
        placement = NoiseAwarePlacement(backend)
        routing = RoutingPass(placement)

        # Optimization sequence
        SequencePass([
            DecomposeBoxes(),
            AutoRebasePass({}),
            routing,
            placement
        ]).apply(tket_circ)

        logger.info(f"✅ TKET Optimization Complete: Depth={tket_circ.depth()}, Gates={tket_circ.n_gates}")
        return tket_circ
    except Exception as e:
        logger.error(f"❌ TKET compilation failed: {e}")
        return kernel

# ===== ERROR MITIGATION =====
def apply_pauli_twirling(kernel, config, dxs, dys, bits, shots):
    """Pauli twirling via multi-run averaging"""
    if not config.USE_PAULI_TWIRLING:
        return Counter()

    logger.info("🌀 Applying Pauli twirling...")
    twirl_counts = Counter()

    for _ in range(4):  # I, X, Y, Z basis averaging
        phase = random.choice([0, math.pi/2, math.pi, 3*math.pi/2])
        logger.debug(f"Twirling run with phase {phase:.2f}")

        raw = emulate_kernel(kernel, bits, dxs, dys) if BACKEND_MODE != "HELIOS" else []
        bitstr = "".join("1" if b else "0" for b in raw)
        twirl_counts[bitstr] += 1

    for bitstr in twirl_counts:
        twirl_counts[bitstr] /= 4
    return twirl_counts

def manual_zne(counts_list: List[Dict[str, int]]) -> Dict[str, int]:
    """Zero-Noise Extrapolation with linear fit"""
    extrapolated = defaultdict(int)
    for bitstr in counts_list[0]:
        vals = [c.get(bitstr, 0) for c in counts_list]
        if len(vals) > 1:
            fit = np.polyfit([1, 3, 5], vals, 1)
            extrapolated[bitstr] = max(0, int(fit[1]))
        else:
            extrapolated[bitstr] = vals[0]
    return extrapolated

# ===== Q-NEXUS FUNCTIONS =====
def check_qnexus_limits():
    """Check Q-Nexus account limits"""
    try:
        import qnexus as qnx
        if not qnx.is_authenticated():
            qnx.login()

        limits = qnx.Account.limits()
        logger.info(f"📋 Q-Nexus Account Limits:")
        logger.info(f"   Max shots per job: {limits.max_shots}")
        logger.info(f"   Daily shot limit: {limits.daily_shots}")
        logger.info(f"   Concurrent jobs: {limits.concurrent_jobs}")
        return limits
    except Exception as e:
        logger.warning(f"⚠️ Could not check Q-Nexus limits: {e}")
        return None

def estimate_cost(qubits: int, shots: int, target: str = "H1-1E"):
    """Estimate job cost"""
    try:
        import qnexus as qnx
        cost = qnx.Job.estimate_cost(
            qubits=qubits,
            shots=shots,
            target=target
        )
        return cost
    except Exception as e:
        logger.warning(f"⚠️ Could not estimate cost: {e}")
        return None

def request_limit_increase(max_shots: int = 1000000, justification: str = "ECDLP research"):
    """Request limit increase"""
    try:
        import qnexus as qnx
        if not qnx.is_authenticated():
            qnx.login()

        response = qnx.Account.request_limit_increase(
            max_shots=max_shots,
            justification=justification
        )
        logger.info(f"📝 Limit increase request submitted. Response: {response}")
        return response
    except Exception as e:
        logger.error(f"❌ Limit increase request failed: {e}")
        return None

def run_multiple_jobs(kernel, config, dxs, dys, total_shots, jobs=5):
    """Split large shot counts into multiple jobs"""
    shots_per_job = total_shots // jobs
    all_counts = Counter()

    for i in range(jobs):
        try:
            counts = submit_to_qnexus(kernel, config, dxs, dys, shots_per_job)
            all_counts.update(counts)
            logger.info(f"✅ Completed job {i+1}/{jobs}")
        except Exception as e:
            logger.error(f"❌ Job {i+1} failed: {e}")
            continue

    return all_counts

def submit_to_qnexus(kernel, config, dxs, dys, shots=None):
    """Submit job to Q-Nexus with proper error handling"""
    try:
        import qnexus as qnx
        if not qnx.is_authenticated():
            qnx.login()

        # Use provided shots or config shots
        actual_shots = shots if shots is not None else config.SHOTS

        # Check if we need to split into multiple jobs
        limits = check_qnexus_limits()
        if limits and actual_shots > limits.max_shots:
            logger.info(f"📋 Shot count {actual_shots} exceeds single job limit of {limits.max_shots}")
            logger.info("🔄 Splitting into multiple jobs...")
            return run_multiple_jobs(kernel, config, dxs, dys, actual_shots)

        project = qnx.Project.get_or_create(name=config.QNEXUS_PROJECT)
        inputs = {
            "bits": config.BITS,
            "dxs": [int(x) for x in dxs],
            "dys": [int(y) for y in dys]
        }

        # Estimate cost before submission
        cost = estimate_cost(config.BITS + 2, actual_shots, config.TARGET_MACHINE)
        if cost is not None:
            logger.info(f"💰 Estimated cost: {cost}")

        logger.info(f"🚀 Submitting to Q-Nexus {config.TARGET_MACHINE}...")
        job = qnx.submit(
            program=kernel,
            inputs=inputs,
            target=qnx.Machine.get(config.TARGET_MACHINE),
            shots=actual_shots,
            project=project,
            options={
                "error_mitigation": {
                    "pec": config.USE_PEC,
                    "zne": config.USE_ZNE,
                    "dd_sequence": config.DD_SEQUENCE if config.USE_DD else None
                }
            }
        )

        logger.info(f"📝 Job submitted: {job.id}")
        logger.info("⏳ Waiting for results...")

        while job.status() not in [JobStatus.COMPLETED, JobStatus.FAILED]:
            time.sleep(10)
            logger.info(f"📊 Job status: {job.status()} - {job.progress()}%")

        if job.status() == JobStatus.FAILED:
            raise RuntimeError(f"❌ Job failed: {job.error()}")

        return job.results().get_counts()

    except Exception as e:
        logger.error(f"❌ Q-Nexus submission failed: {e}")
        raise

# ===== POST-PROCESSING =====
def process_measurement(meas: int, bits: int, order: int) -> List[int]:
    """Process measurement with dual endianness"""
    candidates = []

    # LSB processing
    num, den = Fraction(meas, 1 << bits).limit_denominator(order)
    if den != 0:
        inv = pow(den, -1, order)
        if inv: candidates.append((num * inv) % order)
    candidates.extend([meas % order, (order - meas) % order])

    # MSB processing
    meas_msb = int(f"{meas:0{bits}b}"[::-1], 2)
    num_msb, den_msb = Fraction(meas_msb, 1 << bits).limit_denominator(order)
    if den_msb != 0:
        inv_msb = pow(den_msb, -1, order)
        if inv_msb: candidates.append((num_msb * inv_msb) % order)
    candidates.extend([meas_msb % order, (order - meas_msb) % order])

    return candidates

def bb_correction(measurements: List[int], order: int) -> int:
    """Bivariate Bicycle Correction"""
    best = None
    max_score = 0
    for cand in set(measurements):
        score = sum(1 for m in measurements if math.gcd(m - cand, order) == 1)
        if score > max_score:
            max_score = score
            best = cand
    return best or 0

def verify_key(k: int, target_x: int) -> bool:
    """Verify candidate key against target public key"""
    Pt = ec_scalar_mult(k, G)
    return Pt is not None and Pt.x() == target_x

def save_key(k: int):
    """Save recovered key to file"""
    hex_k = hex(k)[2:].zfill(64)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    fn = f"recovered_key_{ts}.txt"
    with open(fn, "w") as f:
        f.write(f"Private Key: 0x{hex_k}\n")
        f.write(f"Decimal: {k}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Donation Address: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai\n")
    logger.info(f"🔑 Key saved to: {fn}")

# ===== MAIN EXECUTION =====
def main():
    """Main execution flow with complete ECDLP preprocessing"""
    config = Config()
    config.interactive_setup()

    # Calculate qubit requirements
    qubits_needed = config.BITS + 2
    if config.MODE in [42, 99]:
        qubits_needed += 2

    if BACKEND_MODE != "HELIOS" and qubits_needed > 25:
        logger.warning(f"⚠️ ~{qubits_needed} qubits may exceed Selene limits")

    # Prepare target using complete ECDLP preprocessing
    Q = decompress_pubkey(config.PUBKEY_HEX)
    delta, dxs, dys = precompute_target(Q, config.KEYSPACE_START, config.BITS)

    # Select kernel
    kernels = {
        0: mode_0_diagnostic,
        29: mode_29_qpe_omega,
        30: mode_30_geometric_qpe,
        41: mode_41_shor,
        42: mode_42_hive,
        43: mode_43_ft_qpe,
        99: mode_99_best
    }
    kernel = kernels.get(config.MODE, mode_99_best)

    # Apply TKET compilation if enabled
    if config.USE_TKET:
        kernel = compile_with_tket(kernel, config)

    # Execute
    counts_list = []
    for scale in config.ZNE_SCALES if config.USE_ZNE else [1]:
        counts = apply_pauli_twirling(kernel, config, dxs, dys, config.BITS, config.SHOTS)

        if BACKEND_MODE == "HELIOS":
            try:
                counts = submit_to_qnexus(kernel, config, dxs, dys)
            except Exception as e:
                logger.error(f"⚠️ Falling back to local execution: {e}")
                BACKEND_MODE = "SELENE_PYPI"
                counts = Counter()
        else:
            counts = Counter()
            logger.info(f"💻 Executing on {BACKEND_MODE}...")
            for _ in range(config.SHOTS):
                raw = emulate_kernel(kernel, config.BITS, dxs, dys)
                bitstr = "".join("1" if b else "0" for b in raw)
                counts[bitstr] += 1

        counts_list.append(counts)

    # Process results
    if config.USE_ZNE and len(counts_list) > 1:
        final_counts = manual_zne(counts_list)
    else:
        final_counts = counts_list[0]

    measurements = []
    for bitstr, cnt in final_counts.most_common(config.SEARCH_DEPTH):
        val = int(bitstr, 2)
        measurements.extend(process_measurement(val, config.BITS, ORDER))

    filtered = [m for m in measurements if math.gcd(m, ORDER) == 1]
    candidate = bb_correction(filtered, ORDER)

    # Results
    print("\n📊 Results:")
    for cand in sorted(set(filtered), reverse=True)[:10]:
        if verify_key(cand, Q.x()):
            print(f"🔥 SUCCESS → Private key: {hex(cand)}")
            print(f"   A Donation Please: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai 💰")
            save_key(cand)
            return

    print("❌ No key found in top candidates")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.bar(range(len(final_counts)), list(final_counts.values()))
    plt.title("Measurement Distribution")
    plt.xticks(range(0, len(final_counts), max(1, len(final_counts)//10)),
               [hex(int(k, 2))[:10] for k in final_counts.keys()[::max(1, len(final_counts)//10)]],
               rotation=45)

    plt.subplot(2, 1, 2)
    top_cands = sorted(set(filtered), reverse=True)[:20]
    plt.bar(range(len(top_cands)), [1]*len(top_cands))
    plt.title("Top Candidates (Filtered)")
    plt.xticks(range(len(top_cands)), [hex(c)[:10] for c in top_cands], rotation=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("""
    🐉 DRAGON_CODE v135-G Q-Nexus — Quantinuum Guppy ECDLP Solver 🐉
    -----------------------------------------------------------------
    Donation Please: 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai 💰
    -----------------------------------------------------------------
    🚀 Starting Quantinuum Guppy ECDLP Solver...
    """)
    main()
