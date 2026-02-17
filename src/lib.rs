//! # Rug-Calc
//!
//! `rug_calc` is a high-performance, high-precision expression evaluator.
//! It bridges the gap between raw MPFR calculations and user-friendly string expressions.
//!
//! ## Why Rug-Calc?
//!
//! * **High Performance**: Non-recursive implementation, calculation completed in a single traversal.
//! * **Ultra-High Precision**: While standard `f64` is limited to ~15-17 decimal digits,
//!     this engine performs internal calculations with **2560-bit precision**, allowing
//!     for hundreds of accurate decimal places.
//! * **Comprehensive Function Support**: Supports over 40 mathematical functions, including
//!     trigonometry, hyperbolic functions, Gamma, Zeta, and Airy functions.
//! * **Zero-Copy Logic**: Efficiently parses expressions using byte-level scanning and
//!     state-machine logic to minimize overhead.
//!     ensuring safe integration into production systems.
//!
use rug::Float;
use rug::ops::Pow;
use once_cell::sync::Lazy;
use rug::float::Constant;
use phf::phf_map;
use phf::Map;

#[derive(Clone)]
enum Marker {
    Init,
    Number,
    NegSub,
    LParen,
    RParen,
    Char,
    Const,
    Func,
}

#[derive(Clone)]
enum State {
    Initial,
    Operator,
    Operand,
}

/// Errors that can occur during expression parsing or mathematical evaluation.
#[derive(Clone, Debug)]
pub enum CalcError {
    /// The character is not recognized as a valid operator or command.
    UnknownOperator,
    /// A generic container for custom error strings.
    Custom(String),
    /// Division by zero or modulo by zero.
    DivideByZero,
    /// Result is too large or too small for the current precision limits.
    BeyondAccuracy,
    /// An internal engine error that shouldn't normally be reached.
    UnknownError,
    /// The input value is outside the function's domain (e.g., `ln(-1)`).
    ParameterError,
    /// Syntax error in the expression (e.g., unbalanced parentheses).
    ExpressionError,
    /// The specified function name does not exist.
    FunctionUndefined,
    /// The character is not a recognized operator.
    OperatorUndefined,
    /// The expression ended abruptly without an equal sign or newline.
    NoTerminator,
    /// Provided string is empty or contains only whitespace.
    EmptyExpression,
    /// Failed to parse a string literal into a valid floating-point number.
    InvalidNumber,
}

static MAX: Lazy<Float> = Lazy::new(||{
    let max = Float::parse("1e+764").unwrap();
    Float::with_val(2560, max)
});

type MathFn = fn(Float) -> Result<Float, CalcError>;
static MATH: Map<&'static str, MathFn> = phf_map! {
    "ai" => |v| v.ai().accuracy(),
    "li" => |v| v.li2().accuracy(),
    "erf" => |v| v.erf().accuracy(),
    "erfc" => |v| v.erfc().accuracy(),
    "abs" => |v| v.abs().accuracy(),
    "ln" => |v| if v <= 0.0 {
        Err(CalcError::ParameterError)
    } else { v.ln().accuracy() },
    "exp" => |v| v.exp().accuracy(),
    "expt" => |v| v.exp2().accuracy(),
    "expx" => |v| v.exp10().accuracy(),
    "trunc" => |v| v.trunc().accuracy(),
    "zeta" => |v| if v == 1.0 {
        Err(CalcError::ParameterError)
    } else { v.zeta().accuracy() },
    "gamma" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.gamma().accuracy() },
    "digamma" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.digamma().accuracy() },
    "eint" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.eint().accuracy() },
    "log" => |v| if v <= 0.0 {
        Err(CalcError::ParameterError)
    } else { v.log2().accuracy() },
    "logx" => |v| if v <= 0.0 {
        Err(CalcError::ParameterError)
    } else { v.log10().accuracy() },
    "cos" => |v| v.cos().accuracy(),
    "sin" => |v| v.sin().accuracy(),
    "tan" => |v| v.tan().accuracy(),
    "sec" => |v| v.sec().accuracy(),
    "csc" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.csc().accuracy() },
    "cot" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.cot().accuracy() },
    "cosh" => |v| v.cosh().accuracy(),
    "sinh" => |v| v.sinh().accuracy(),
    "tanh" => |v| v.tanh().accuracy(),
    "ceil" => |v| v.ceil().accuracy(),
    "floor" => |v| v.floor().accuracy(),
    "frac" => |v| v.fract().accuracy(),
    "sgn" => |v| v.signum().accuracy(),
    "recip" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.recip().accuracy() },
    "csch" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.csch().accuracy() },
    "sech" => |v| v.sech().accuracy(),
    "coth" => |v| if v == 0.0 {
        Err(CalcError::ParameterError)
    } else { v.coth().accuracy() },
    "acos" => |v| if v < -1.0 || v > 1.0 {
        Err(CalcError::ParameterError)
    } else { v.acos().accuracy() },
    "asin" => |v| if v < -1.0 || v > 1.0 {
        Err(CalcError::ParameterError)
    } else { v.asin().accuracy() },
    "atan" => |v| v.atan().accuracy(),
    "acosh" => |v| if v < 1.0 {
        Err(CalcError::ParameterError)
    } else { v.acosh().accuracy() },
    "asinh" => |v| v.asinh().accuracy(),
    "atanh" => |v| if v <= -1.0 || v >= 1.0 {
        Err(CalcError::ParameterError)
    } else { v.atanh().accuracy() },
    "cbrt" => |v| v.cbrt().accuracy(),
    "sqrt" => |v| if v < 0.0 {
        Err(CalcError::ParameterError)
    } else { v.sqrt().accuracy() },
    "fac" => |v| {
        let to_u32 = v.to_u32_saturating().unwrap();
        let fac = Float::factorial(to_u32);
        Float::with_val(2560, fac).accuracy()
    },
};

/// The core Calculation engine.
///
/// It maintains internal buffers for operators, operands, and function pointers
/// to minimize allocations during repeated evaluations.
///
/// # Example
/// ```
/// use rug_calc::Calculator;
/// let mut calc = Calculator::new();
/// let val = calc.run(&"sqrt(2)".to_string()).unwrap();
/// ```
#[derive(Clone)]
pub struct Calculator {
    marker: Marker,
    operator: Vec<u8>,
    function: Vec<Option<MathFn>>,
    numbers: Vec<Float>,
    state: State,
}

/// Extension trait for `u8` to handle operator logic.
trait ByteExt {
    /// Returns the precedence of the operator. Higher values mean higher priority.
    fn priority(&self) -> Result<u8, CalcError>;
    /// Performs the mathematical operation using the top two numbers from the calculator stack.
    fn computing(&self, n: &mut Calculator) -> Result<Float, CalcError>;
}

/// Extension trait for `rug::Float` to provide formatting and accuracy checks.
trait FloatExt {
    /// Floating-point modulo operation.
    fn fmod(&self, n: &Float) -> Float;
    /// Validates if the result is a finite number and within the allowed magnitude.
    fn accuracy(self) -> Result<Float, CalcError>;
    /// Converts the high-precision float to a rounded string representation.
    /// Supports custom precision up to 700 decimal places.
    fn to_round(&self, digits: Option<usize>) -> Result<String, CalcError>;
}

/// Extension trait for `String` to facilitate raw data parsing and formatting.
trait StringExt {
    /// Parses a raw Rug Float string into its components (sign, digits, exponent).
    fn parse_rug_raw(&self) -> (bool, Vec<u8>, i32);
    /// Formats a float string into a fixed-point representation without trailing zeros.
    fn to_fixed_clean(&self) -> Result<String, CalcError>;
    /// Rounds the floating-point string to a specific number of decimal places.
    fn to_fixed_round(&self, prec: i32) -> Result<String, CalcError>;
    /// Extracts a number from a substring and converts it into a high-precision `Float`.
    fn extract(&self, n: usize, i: usize) -> Result<Float, CalcError>;
}

impl ByteExt for u8 {
    fn priority(&self) -> Result<u8, CalcError> {
        match self {
            b'+' | b'-' => Ok(1),
            b'*' | b'/' | b'%' => Ok(2),
            b'^' => Ok(3),
            _ => Err(CalcError::UnknownOperator)
        }
    }

    fn computing(&self, num: &mut Calculator) -> Result<Float, CalcError> {
        let c1 = num.numbers.pop().ok_or(CalcError::ExpressionError)?;
        let c2 = num.numbers.pop().ok_or(CalcError::ExpressionError)?;
        match self {
            b'+' => Float::with_val(2560, &c2 + &c1).accuracy(),
            b'-' => Float::with_val(2560, &c2 - &c1).accuracy(),
            b'*' => Float::with_val(2560, &c2 * &c1).accuracy(),
            b'/' if &c1 != &0.0 => Float::with_val(2560, &c2 / &c1).accuracy(),
            b'%' if &c1 != &0.0 => c2.fmod(&c1).accuracy(),
            b'^' => Float::with_val(2560, &c2.pow(&c1)).accuracy(),
            _ => Err(CalcError::DivideByZero)
        }
    }
}

impl FloatExt for Float {
    fn fmod(&self, n: &Float) -> Float {
        let mut m = Float::with_val(2560, self / n);
        if self < &0.0 {
            m.ceil_mut()
        } else { m.floor_mut() };
        Float::with_val(2560, self - &m * n)
    }

    fn accuracy(self) -> Result<Float, CalcError> {
        if self.is_nan() || self.is_infinite() {
            Err(CalcError::BeyondAccuracy)
        } else if self > *MAX || self < *MAX.as_neg() {
            Err(CalcError::BeyondAccuracy)
        } else { Ok(self) }
    }

    fn to_round(&self, digits: Option<usize>) -> Result<String, CalcError> {
        if let Some(precision) = digits {
            if precision < 1 || precision > 700 {
                let err = String::from("Set Precision Greater Than Equal 1");
                return Err(CalcError::Custom(err));
            }
        }
        let raw = self.to_string_radix(10, None);
        match digits {
            None => raw.to_fixed_clean(),
            Some(digits) => raw.to_fixed_round(digits as i32),
        }
    }
}

impl StringExt for String {
    fn parse_rug_raw(&self) -> (bool, Vec<u8>, i32) {
        let bytes = self.as_bytes();
        let is_neg = bytes.starts_with(&[b'-']);
        let start = if is_neg { 1 } else { 0 };
        let e_pos = bytes.iter().position(|&b| b == b'e');
        let end = e_pos.unwrap_or(bytes.len());
        let mantissa = &bytes[start..end];
        let mut digits = Vec::with_capacity(mantissa.len());
        let mut dot_pos = None;
        for (index, &byte) in mantissa.iter().enumerate() {
            if byte == b'.' {
                dot_pos = Some(index as i32);
            } else { digits.push(byte); }
        }
        let raw_exp: i32 = e_pos.map(|pos|{
            self[pos+1..].parse().unwrap_or(0)
        }).unwrap_or(0);
        let adj_exp = dot_pos.map(|pos| raw_exp+pos)
        .unwrap_or(raw_exp+digits.len() as i32);
        (is_neg, digits, adj_exp)
    }

    fn to_fixed_clean(&self) -> Result<String, CalcError> {
        let (negative, digits, exp) = self.parse_rug_raw();
        let mut cursor = 0;
        let digits_len = digits.len();
        let exp_abs = (exp.unsigned_abs()+2) as usize;
        let mut buf = vec![b'0'; digits_len+exp_abs];
        if negative {
            buf[cursor] = b'-';
            cursor += 1;
        }
        let dot_pos: Option<usize>;
        if exp <= 0 {
            buf[cursor..cursor+2]
            .copy_from_slice(b"0.");
            dot_pos = Some(cursor+1);
            cursor += 2;
            let zeros = exp.abs() as usize;
            cursor += zeros;
            buf[cursor..cursor+digits_len]
            .copy_from_slice(&digits);
            cursor += digits_len;
        } else {
            let dot_idx = exp as usize;
            if dot_idx >= digits_len {
                buf[cursor..cursor+digits_len]
                .copy_from_slice(&digits);
                cursor += digits_len;
                let trailing_zeros = dot_idx-digits_len;
                cursor += trailing_zeros;
                dot_pos = None;
            } else {
                buf[cursor..cursor+dot_idx]
                .copy_from_slice(&digits[..dot_idx]);
                cursor += dot_idx;
                buf[cursor] = b'.';
                dot_pos = Some(cursor);
                cursor += 1;
                let rem = digits_len-dot_idx;
                buf[cursor..cursor+rem]
                .copy_from_slice(&digits[dot_idx..]);
                cursor += rem;
            }
        }
        let mut final_len = cursor;
        if let Some(dot) = dot_pos {
            let dec_len = cursor-dot;
            final_len = if dec_len < 700
            { dec_len } else { 700 };
            while final_len > 0 {
                match buf[final_len-1] {
                    b'0' => final_len -= 1,
                    b'.' => { final_len -= 1; break; }
                    _ => break,
                }
            }
        }
        buf.truncate(final_len);
        Ok(String::from_utf8(buf).unwrap())
    }

    fn to_fixed_round(&self, prec: i32) -> Result<String, CalcError> {
        let (negative, digits, exp) = self.parse_rug_raw();
        let round_idx = (exp+prec) as usize;
        let mut carry = false;
        if round_idx < digits.len() && digits[round_idx] >= b'5' {
            carry = true;
        }
        let mut buf = [b'0'; 4096];
        let mut cursor = 4095;
        let max_bound = (exp+prec)-1;
        let min_bound = std::cmp::min(0, exp-1);
        for index in (min_bound..=max_bound).rev() {
            if index == exp-1 && prec > 0 {
                buf[cursor] = b'.';
                cursor -= 1;
            }
            let index = index as usize;
            let mut digit = if index < digits.len() {
                digits[index]
            } else { b'0' };
            if carry {
                if digit == b'9' {
                    digit = b'0';
                } else {
                    digit += 1;
                    carry = false;
                }
            }
            buf[cursor] = digit;
            cursor -= 1;
        }
        if carry {
            buf[cursor] = b'1';
            cursor -= 1;
        }
        if negative {
            buf[cursor] = b'-';
            cursor -= 1;
        }
        let final_buf = &buf[cursor+1..4096];
        let mut final_len = final_buf.len();
        for index in final_buf.iter().rev() {
            match index {
                b'0' => final_len -= 1,
                b'.' => { final_len -= 1; break; }
                _ => break,
            }
        }
        let result = final_buf[..final_len].to_vec();
        Ok(String::from_utf8(result).unwrap())
    }

    fn extract(&self, n: usize, i: usize) -> Result<Float, CalcError> {
        match Float::parse(&self[n..i]) {
            Ok(valid) => Float::with_val(2560, valid).accuracy(),
            Err(_) => Err(CalcError::InvalidNumber)
        }
    }
}

impl CalcError {
    pub fn to_string(&self) -> String {
        match self {
            CalcError::UnknownOperator => String::from("Unknown Operator"),
            CalcError::Custom(error) => String::from(error),
            CalcError::DivideByZero => String::from("Divide By Zero"),
            CalcError::BeyondAccuracy => String::from("Beyond Accuracy"),
            CalcError::UnknownError => String::from("Unknown Error"),
            CalcError::ParameterError => String::from("Parameter Error"),
            CalcError::ExpressionError => String::from("Expression Error"),
            CalcError::FunctionUndefined => String::from("Function Undefined"),
            CalcError::OperatorUndefined => String::from("Operator Undefined"),
            CalcError::NoTerminator => String::from("No Terminator"),
            CalcError::EmptyExpression => String::from("Empty Expression"),
            CalcError::InvalidNumber => String::from("Invalid Number"),
        }
    }
}

impl Calculator {
    /// Initializes a new Calculator instance with pre-allocated capacities.
    pub fn new() -> Self {
        Self {
            state: State::Initial,
            numbers: Vec::with_capacity(32),
            function: vec![None; 32],
            operator: Vec::with_capacity(32),
            marker: Marker::Init,
        }
    }

    /// Resets the internal state and clears all stacks.
    /// Useful for reusing the same instance for a new calculation.
    pub fn reset(&mut self) {
        self.numbers.clear();
        self.state = State::Initial;
        self.function.fill(None);
        self.marker = Marker::Init;
        self.operator.clear();
    }

    /// Evaluates a mathematical expression and returns the raw `rug::Float` result.
    ///
    /// # Supported Syntax
    /// - Basic: `+`, `-`, `*`, `/`, `%`, `^`
    /// - Functions: `sin(x)`, `ln(x)`, `fac(x)` (factorial), etc.
    /// - Constants: `P` (Pi), `E` (Euler), `C` (Catalan), `L` (Log2)
    ///
    /// # Examples
    /// ```
    /// let mut calc = Calculator::new();
    /// let res = calc.run(&"sin(P/2)".to_string()).unwrap();
    /// ```
    pub fn run(&mut self, expr: &String) -> Result<Float, CalcError> {
        let expr = format!("{}=", expr);
        let mut locat: usize = 0;
        let mut bracket: usize = 0;

        for (index, &valid) in expr.as_bytes().iter().enumerate() {
            match valid {
                b'0'..=b'9' | b'.' => {
                    if !matches!(self.marker, Marker::RParen | Marker::Const | Marker::Func) {
                        self.marker = Marker::Number;
                        continue;
                    }
                    return Err(CalcError::ExpressionError);
                }

                b'a'..=b'z' => {
                    if !matches!(self.marker, Marker::RParen | Marker::Const | Marker::NegSub | Marker::Number) {
                        self.marker = Marker::Func;
                        continue;
                    }
                    return Err(CalcError::ExpressionError);
                }

                ch @ b'+' | ch @ b'-' | ch @ b'*' | ch @ b'/' | ch @ b'%' | ch @ b'^' => {
                    if ch == b'-' && matches!(self.marker, Marker::Init | Marker::LParen | Marker::Char) {
                        self.marker = Marker::NegSub;
                        continue;
                    } else if !matches!(self.marker, Marker::Number | Marker::RParen | Marker::Const) {
                        return Err(CalcError::ExpressionError);
                    }

                    if matches!(self.state, State::Operator | State::Initial) {
                        self.numbers.push(expr.extract(locat, index)?);
                        self.state = State::Operand;
                    }

                    while self.operator.len() != 0 && self.operator.last().unwrap() != &b'(' {
                        if self.operator.last().unwrap().priority()? >= ch.priority()? {
                            let value = self.operator.pop().unwrap().computing(self)?;
                            self.numbers.push(value);
                        } else {
                            break;
                        }
                    }

                    self.operator.push(ch);
                    self.state = State::Operator;
                    self.marker = Marker::Char;
                    locat = index + 1;
                    continue;
                }

                ch @ b'(' => {
                    if matches!(self.marker, Marker::Func) {
                        let name = &expr[locat..index];
                        if let Some(&func_ptr) = MATH.get(name) {
                            self.function[bracket+1] = Some(func_ptr);
                        } else {
                            return Err(CalcError::FunctionUndefined);
                        }
                    }

                    if matches!(self.state, State::Operator | State::Initial) {
                        if !matches!(self.marker, Marker::Number | Marker::NegSub) {
                            self.operator.push(ch);
                            locat = index + 1;
                            self.marker = Marker::LParen;
                            bracket += 1;
                            continue;
                        }
                    }
                    return Err(CalcError::ExpressionError);
                }

                b')' => {
                    if matches!(self.state, State::Operator | State::Initial) {
                        if matches!(self.marker, Marker::Number) {
                            self.numbers.push(expr.extract(locat, index)?);
                            self.state = State::Operand;
                        }
                    }

                    if matches!(self.state, State::Operand) {
                        if bracket > 0 {
                            while self.operator.last().unwrap() != &b'(' {
                                let value = self.operator.pop().unwrap().computing(self)?;
                                self.numbers.push(value);
                            }

                            if let Some(func) = self.function[bracket].take() {
                                let value = self.numbers.pop().unwrap();
                                self.numbers.push(func(value)?);
                            }

                            locat = index + 1;
                            self.operator.pop();
                            self.marker = Marker::RParen;
                            bracket -= 1;
                            continue;
                        }
                    }
                    return Err(CalcError::ExpressionError);
                }

                b'=' | b'\n' | b'\r' => {
                    if matches!(self.marker, Marker::Init) {
                        return Err(CalcError::EmptyExpression);
                    } else if bracket > 0 || matches!(self.marker, Marker::NegSub | Marker::Char | Marker::Func) {
                        return Err(CalcError::ExpressionError);
                    }

                    if matches!(self.state, State::Operator | State::Initial) {
                        self.numbers.push(expr.extract(locat, index)?);
                        self.state = State::Operand;
                    }

                    while self.operator.len() != 0 {
                        let value = self.operator.pop().unwrap().computing(self)?;
                        self.numbers.push(value);
                    }

                    let result = self.numbers.pop().unwrap();
                    self.reset(); return Ok(result);
                }

                ch @ b'P' | ch @ b'E' | ch @ b'C' | ch @ b'L' => {
                    if matches!(self.state, State::Operator | State::Initial) {
                        let constant = match ch {
                            b'P' => &Constant::Pi,
                            b'E' => &Constant::Euler,
                            b'C' => &Constant::Catalan,
                            b'L' => &Constant::Log2,
                            _ => return Err(CalcError::UnknownError)
                        };

                        if !matches!(self.marker, Marker::Number | Marker::Func) {
                            let value = if matches!(self.marker, Marker::NegSub) {
                                0.0 - Float::with_val(128, constant)
                            } else {
                                Float::with_val(128, constant)
                            };
                            self.numbers.push(value);
                            self.state = State::Operand;
                            self.marker = Marker::Const;
                            locat = index + 1;
                            continue;
                        }
                    }
                    return Err(CalcError::ExpressionError);
                }

                _ => return Err(CalcError::OperatorUndefined)
            }
        }
        Err(CalcError::NoTerminator)
    }

    /// Evaluates an expression and returns a formatted string.
    ///
    /// # Arguments
    /// * `digits` - Number of decimal places to round to. Supports 1 to 700.
    ///
    /// # Example
    /// ```
    /// let mut calc = Calculator::new();
    /// let s = calc.run_round(&"1/3".to_string(), Some(5)).unwrap();
    /// assert_eq!(s, "0.33333");
    /// ```
    pub fn run_round(
        &mut self, expr: &String, digits: Option<usize>
    ) -> Result<String, CalcError> {
        match self.run(expr) {
            Ok(value) => Ok(value.to_round(digits)?),
            Err(err) => Err(err)
        }
    }
}
