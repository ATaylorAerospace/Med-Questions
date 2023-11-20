import numpy as np
import math
import matplotlib.pyplot as plt

# Functions to quantize and unquantize
def quantize(value, bits=4):
    quantized_value = np.round(value * (2**(bits - 1) - 1))
    return int(quantized_value)

def unquantize(quantized_value, bits=4):
    value = quantized_value / (2**(bits - 1) - 1)
    return float(value)

quant_4 = quantize(0.622, 4)
print(quant_4)

quant_8 = quantize(0.622, 8)
print(quant_8)

unquant_4 = unquantize(quant_4, 4)
print(unquant_4)

unquant_8 = unquantize(quant_8, 8)
print(unquant_8)

# Original numbers: 0.6222
# Unquantized values: 0.571 for 4-bit and 0.622 for 8-bit quantization

x = np.linspace(-1, 1, 50)
y = [math.cos(val) for val in x]

y_quant_8bit = np.array([quantize(val, bits=8) for val in y])
y_unquant_8bit = np.array([unquantize(val, bits=8) for val in y_quant_8bit])

y_quant_4bit = np.array([quantize(val, bits=4) for val in y])
y_unquant_4bit = np.array([unquantize(val, bits=4) for val in y_quant_4bit])

plt.figure(figsize=(10, 12))

plt.subplot(4, 1, 1)
plt.plot(x, y, label="Original")
plt.plot(x, y_unquant_8bit, label="Unquantized_8bit")
plt.plot(x, y_unquant_4bit, label="Unquantized_4bit")
plt.legend()
plt.title("Comparison Graph")
plt.grid(True)

plt.show()
