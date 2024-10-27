
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
