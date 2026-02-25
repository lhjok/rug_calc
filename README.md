# 🧮 Rug-Calc

A zero-copy, high-performance evaluator for arbitrary-precision scientific computing. Powered by Rug, 
it delivers extreme accuracy and numerical stability with minimal memory overhead.

## ✨ Features

* 🚀 **High Performance**: Features a **non-recursive** architecture that evaluates expressions in a **single traversal**, ensuring predictable and blazing-fast execution.
* ⚖️ **Arbitrary Precision**: Engineered with **2560-bit internal precision**, far exceeding the limitations of standard hardware floats for mission-critical calculations.
* 📦 **Zero-Copy & Efficient**: Utilizes **byte-level scanning** and a custom **state-machine** to minimize memory allocations and CPU overhead during expression parsing.
* 🧪 **Rich Mathematical Suite**: Supports **40+ built-in functions**, including advanced Trigonometry, Gamma, Zeta, Airy, and Error functions.
* 🔭 **Scientific Constants**: Instant, high-precision access to fundamental constants: Pi (**P**), Euler (**Y**), Catalan (**C**), and Log2 (**L**).
* 🎨 **Smart Formatting**: Provides versatile output options, including **fixed-point rounding** and **"clean" string formatting** with support for up to 700 decimal places.
* 🛡️ **Robust Stability**: Designed for applications requiring **extreme numerical stability** and graceful error handling over panics.

## 🚀 Quick Start
Add this to your `Cargo.toml`:

```toml
[dependencies]
rug_calc = "0.1.6"
```

## 📚 Mathematical Function Support
`rug_calc` provides a comprehensive suite of high-precision functions powered by the MPFR library.
* `ai` , `abs` , `cos` , `sin` , `tan` , `csc` , `sec` , `cot` , `coth` , `ceil`
* `cosh` , `sinh` , `tanh` , `sech` , `ln` , `csch` , `acos` , `asin` , `atan`
* `acosh` , `asinh` , `atanh` , `log2` , `log10` , `sqrt` , `cbrt` , `fac` , `recip`
* `erf` , `li2` , `exp` , `exp2` ,`exp10` , `eint` , `zeta` , `trunc` , `gamma`
* `floor` , `frac` , `sgn` , `erfc` , `digamma`

## 💎 Constant Identifiers
To maintain parsing efficiency and avoid ambiguity with functions, constants use single-character uppercase identifiers:
* `P`: Pi constant
* `Y`: Euler-Mascheroni constant
* `C`: Catalan’s constant
* `L`: Natural logarithm of 2 (Log2)

## 💡 Example Usage
The following example demonstrates how to use

```rust
use rug_calc::Calculator;

let mut calc = Calculator::new();

// 1. Simple arithmetic with negative results
let result = calc.run("8*(6+6/2-2*5)+-2").unwrap();

// 2. Scientific Notation (Zero-Copy parsing of 'e')
// Calculates: (1.23*10^-5)*2
let sci_result = calc.run("1.23e-5*2").unwrap();
println!("Scientific result: {}", sci_result);

// 3. Advanced functions (Calculating cos(sin(π/4)))
let cos_sin_pi_4 = calc.run("cos(sin(P/4))").unwrap();

// 4. Mixing Scientific Notation with Special Constants
// Calculates: 5.0*10^12 divided by Euler's constant (Y)
let mixed_result = calc.run("5.0e+12/Y").unwrap();

// 5. Infinitely nested scientific computing
let complex = calc.run("8*6-(cos(6-3*(6/P^2-6)*3)+5)/Y*8").unwrap();

// 6. High-precision string output (50 decimal places)
let pi_str = calc.run_round("P", Some(50)).unwrap();
println!("Pi to 50 places: {}", pi_str);

// 7. Small number formatting from scientific notation
let small_val = calc.run_round("1E-10", Some(12)).unwrap();
println!("Small value: {}", small_val); // "0.0000000001"
```

## Contributing
Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/lhjok/rug_calc).

## License
This project is licensed under the MIT License.