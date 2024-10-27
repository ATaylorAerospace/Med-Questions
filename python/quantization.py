# import necessary libraries
import numpy as np
import math
import matplotlib.pyplot as plt

# Functions to quantize and unquantize
def quantize(value, bits=4):
    # Ensure value is within the range [-1, 1]
    value = max(min(value, 1), -1)
    quantized_value = np.round(value * (2**(bits - 1) - 1))
    return int(quantized_value)

def unquantize(quantized_value, bits=4):
    value = quantized_value / (2**(bits - 1) - 1)
    return float(value)

# Quantize and unquantize example values
quant_4 = quantize(0.622, 4)
print(f"4-bit Quantized: {quant_4}")

quant_8 = quantize(0.622, 8)
print(f"8-bit Quantized: {quant_8}")

unquant_4 = unquantize(quant_4, 4)
print(f"4-bit Unquantized: {unquant_4}")

unquant_8 = unquantize(quant_8, 8)
print(f"8-bit Unquantized: {unquant_8}")

# Generate x values and compute the original y values
x = np.linspace(-1, 1, 50)
y = [math.cos(val) for val in x]

# Quantize and unquantize y values with 8-bit and 4-bit quantization
y_quant_8bit = np.array([quantize(val, bits=8) for val in y])
y_unquant_8bit = np.array([unquantize(val, bits=8) for val in y_quant_8bit])

y_quant_4bit = np.array([quantize(val, bits=4) for val in y])
y_unquant_4bit = np.array([unquantize(val, bits=4) for val in y_quant_4bit])

# Plotting the original and unquantized values
plt.figure(figsize=(10, 8))

plt.plot(x, y, label="Original", color='b')
plt.plot(x, y_unquant_8bit, label="Unquantized 8-bit", linestyle='--', color='g')
plt.plot(x, y_unquant_4bit, label="Unquantized 4-bit", linestyle=':', color='r')

plt.legend()
plt.title("Comparison of Original and Unquantized Values after Quantization")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
