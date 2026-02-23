# 🧮 Rug-Calc

A zero-copy, high-performance evaluator for arbitrary-precision scientific computing. Powered by Rug, 
it delivers extreme accuracy and numerical stability with minimal memory overhead.

## ✨ Features

* 🚀 **High Performance**: Features a **non-recursive** architecture that evaluates expressions in a **single traversal**, ensuring predictable and blazing-fast execution.
* ⚖️ **Arbitrary Precision**: Engineered with **2560-bit internal precision**, far exceeding the limitations of standard hardware floats for mission-critical calculations.
* 📦 **Zero-Copy & Efficient**: Utilizes **byte-level scanning** and a custom **state-machine** to minimize memory allocations and CPU overhead during expression parsing.
* 🧪 **Rich Mathematical Suite**: Supports **40+ built-in functions**, including advanced Trigonometry, Gamma, Zeta, Airy, and Error functions.
* 🔭 **Scientific Constants**: Instant, high-precision access to fundamental constants: Pi (**P**), e (**E**), Catalan (**C**), and ln 2 (**L**).
* 🎨 **Smart Formatting**: Provides versatile output options, including **fixed-point rounding** and **"clean" string formatting** with support for up to 700 decimal places.
* 🛡️ **Robust Stability**: Designed for applications requiring **extreme numerical stability** and graceful error handling over panics.

## 🚀 Quick Start
Add this to your `Cargo.toml`:

```toml
[dependencies]
rug_calc = "0.1.4"
```

## ✨ Performance Tips
### Zero-Copy Parsing
`rug_calc` is designed for high-performance scenarios. To achieve **Zero-Copy** parsing, the engine utilizes a state-machine that expects a terminator to trigger the final calculation.

While the engine can automatically handle expressions without a terminator, it will involve a minor memory allocation to append one internally.

**For maximum performance (Zero-Copy), it is recommended to append an `=` at the end of your expressions:**

## ✨ Example Usage
The following example demonstrates how to use

```rust
use rug_calc::Calculator;

let mut calc = Calculator::new();

// Simple arithmetic
let result = calc.run("8*(6+6/2-2*5)+-2=").unwrap();

// Advanced functions (Calculating cos(sin(π/4)))
let cos_sin_pi_4 = calc.run("cos(sin(P/4))=").unwrap();

// infinitely nested scientific computing
let result = calc.run("8*6-(cos(6-3*(6/P^2-6)*3)+5)/E*8").unwrap();

// High-precision string output (50 decimal places)
let pi_str = calc.run_round("P", Some(50)).unwrap();
println!("Pi to 50 places: {}", pi_str);
```

## Contributing
Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/lhjok/rug_calc).

## License
This project is licensed under the MIT License.