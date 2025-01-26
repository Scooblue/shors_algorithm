from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, transpile
from qiskit.circuit.library import RZGate
import matplotlib.pyplot as plt
from qiskit_aer import Aer
import math

a = 2
N = 15

qBit = math.ceil(math.log2(N))

backend = Aer.get_backend('qasm_simulator')

ip = QuantumRegister(qBit, name='input')
op = QuantumRegister(qBit, name='output')
ar = AncillaRegister(qBit, name='ancilla')
classical = ClassicalRegister(qBit, name='classical')

shorsCircuit = QuantumCircuit(ip, op, ar, classical)

# Hadamard gate / superimpose input register
for qubit in ip:
    shorsCircuit.h(qubit)


def controlledAddition(qc, a, outputReg, controlQubit, ancillaReg):
    n = len(outputReg)
    # Convert a value into binary
    binary_a = bin(a)[2:].zfill(n)

    for i, bit in enumerate(reversed(binary_a)):
        if bit == '1':
            qc.cx(controlQubit, ancillaReg[i])
            qc.cx(ancillaReg[i], outputReg[i])
            qc.cx(controlQubit, ancillaReg[i])


def modularReduction(qc, N, outputReg, ancillaReg):
    n = len(outputReg)
    # Convert N to binary
    binaryN = bin(N)[2:].zfill(n)

    for i, bit in enumerate(reversed(binaryN)):
        if bit == '1':
            qc.cx(outputReg[i], ancillaReg[i])

    for i, bit in enumerate(reversed(binaryN)):
        if bit == '1':
            qc.cx(ancillaReg[-1], outputReg[i])

    for i, bit in enumerate(reversed(binaryN)):
        if bit == '1':
            qc.cx(outputReg[i], ancillaReg[i])


def modularMultiplier(qc, a, N, inputReg, outputReg, ancillaReg):
    """
    Modular multiplier circuit.
    Implements (a * x) % N on quantum registers.
    """
    n = len(inputReg)

    for i in range(n):
        shifted_a = (a * (2 ** i)) % N
        controlledAddition(qc, shifted_a, outputReg, inputReg[i], ancillaReg)
        modularReduction(qc, N, outputReg, ancillaReg)


def modularE(qc, base, modular, inputReg, outputReg):
    numQubits = len(inputReg)
    for i in range(numQubits):
        exponent = base ** (2 ** i)
        modularMultiplier(qc, exponent, modular, inputReg, outputReg, ar)


# Controlled Rk gate (R_n)
def controlledR(qc, qreg, i):
    numQubits = len(qreg)
    for j in range(i + 1, numQubits):
        theta = 2 * 3.14159 / (2 ** (j - i + 1))
        qc.append(RZGate(theta).control(1), [qreg[i], qreg[j]])


def applyQft(qc, register):
    for i in range(len(register)):
        qc.h(register[i])
        controlledR(qc, register, i)
    for i in range(len(register) // 2):
        qc.swap(register[i], register[-i - 1])


shorsCircuit.x(op[0])

modularE(shorsCircuit, base=a, modular=N, inputReg=ip, outputReg=op)

applyQft(shorsCircuit, ip)

shorsCircuit.measure(ip, classical)

shorsCircuit.draw(output='mpl')
plt.show()


def gcd(a, b):
    while b != 0:
        t = b
        b = a % b
        a = t
    return a


def coprime(a, b):
    return gcd(a, b) == 1


def postProcess(counts, base, modular):
    for outcome, freq in counts.items():
        decimal_value = int(outcome, 2)
        fraction = decimal_value / (2 ** len(outcome))

        if fraction == 0:
            continue

        period_guess = round(1 / fraction)

        if period_guess % 2 == 0:
            factor1 = gcd(base ** (period_guess // 2) - 1, modular)
            factor2 = gcd(base ** (period_guess // 2) + 1, modular)

            if factor1 * factor2 == modular and coprime(factor1, factor2) and factor1 != N and factor2 != N:
                print(f"Best Factors found: {factor1}, {factor2}")


transpiled_circuit = transpile(shorsCircuit, backend)
job = backend.run(transpiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts()
print("Measurement Results:", counts)
postProcess(counts, a, N)


def classicalPeriod(modular, base):
    if modular == 1:
        return 0
    r = 1
    if coprime(modular, base):
        while pow(base, r, modular) != 1:
            r += 1
        return r
    else:
        return -1
