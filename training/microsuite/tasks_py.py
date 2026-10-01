"""Original (MIT, FlySlop-authored) Python micro-programs. Each: id, difficulty 1-3, prompt, code, tests."""
_T = []


def _t(id, diff, prompt, code, tests):
    _T.append({"id": f"py/{id}", "lang": "py", "difficulty": diff, "prompt": prompt,
               "code": code.strip("\n") + "\n", "tests": tests.strip("\n") + "\n"})


_t("const5", 1, "Write a function five() that returns 5.", "def five():\n    return 5", "assert five() == 5")
_t("double", 1, "Write a function double(x) that returns x times two.", "def double(x):\n    return x * 2",
   "assert double(3) == 6\nassert double(-2) == -4\nassert double(0) == 0")
_t("triple", 1, "Write a function triple(x) that returns three times x.", "def triple(x):\n    return x * 3",
   "assert triple(4) == 12\nassert triple(-1) == -3")
_t("inc", 1, "Write a function inc(x) that returns x plus one.", "def inc(x):\n    return x + 1",
   "assert inc(0) == 1\nassert inc(-5) == -4")
_t("square", 1, "Write a function square(x) that returns x squared.", "def square(x):\n    return x * x",
   "assert square(5) == 25\nassert square(-3) == 9")
_t("add", 1, "Write a function add(a, b) that returns the sum of a and b.", "def add(a, b):\n    return a + b",
   "assert add(2, 3) == 5\nassert add(-1, 1) == 0")
_t("sub", 1, "Write a function sub(a, b) that returns a minus b.", "def sub(a, b):\n    return a - b",
   "assert sub(5, 3) == 2\nassert sub(0, 4) == -4")
_t("mul", 1, "Write a function mul(a, b) that returns a times b.", "def mul(a, b):\n    return a * b",
   "assert mul(3, 4) == 12\nassert mul(-2, 5) == -10")
_t("neg", 1, "Write a function neg(x) that returns the negation of x.", "def neg(x):\n    return -x",
   "assert neg(3) == -3\nassert neg(-7) == 7\nassert neg(0) == 0")
_t("max2", 1, "Write a function max2(a, b) that returns the larger of a and b.",
   "def max2(a, b):\n    if a > b:\n        return a\n    return b",
   "assert max2(1, 2) == 2\nassert max2(5, 3) == 5\nassert max2(4, 4) == 4")
_t("min2", 1, "Write a function min2(a, b) that returns the smaller of a and b.",
   "def min2(a, b):\n    if a < b:\n        return a\n    return b",
   "assert min2(1, 2) == 1\nassert min2(5, 3) == 3")
_t("absval", 1, "Write a function absval(x) that returns the absolute value of x.",
   "def absval(x):\n    if x < 0:\n        return -x\n    return x",
   "assert absval(-4) == 4\nassert absval(4) == 4\nassert absval(0) == 0")
_t("is_even", 1, "Write a function is_even(n) that returns True if n is even, otherwise False.",
   "def is_even(n):\n    return n % 2 == 0",
   "assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True")
_t("is_odd", 1, "Write a function is_odd(n) that returns True if n is odd, otherwise False.",
   "def is_odd(n):\n    return n % 2 == 1", "assert is_odd(3) is True\nassert is_odd(8) is False")
_t("is_pos", 1, "Write a function is_pos(x) that returns True if x is greater than zero.",
   "def is_pos(x):\n    return x > 0", "assert is_pos(1)\nassert not is_pos(0)\nassert not is_pos(-1)")
_t("sign", 1, "Write a function sign(x) that returns 1 for positive x, -1 for negative x and 0 for zero.",
   "def sign(x):\n    if x > 0:\n        return 1\n    if x < 0:\n        return -1\n    return 0",
   "assert sign(9) == 1\nassert sign(-9) == -1\nassert sign(0) == 0")
_t("clamp", 2, "Write a function clamp(x, lo, hi) that limits x to the range lo to hi.",
   "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
   "assert clamp(5, 0, 10) == 5\nassert clamp(-1, 0, 10) == 0\nassert clamp(11, 0, 10) == 10")
_t("max3", 2, "Write a function max3(a, b, c) that returns the largest of three numbers.",
   "def max3(a, b, c):\n    m = a\n    if b > m:\n        m = b\n    if c > m:\n        m = c\n    return m",
   "assert max3(1, 2, 3) == 3\nassert max3(3, 2, 1) == 3\nassert max3(2, 9, 1) == 9")
_t("sum_list", 2, "Write a function sum_list(xs) that returns the sum of the numbers in the list xs.",
   "def sum_list(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total",
   "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0")
_t("product_list", 2, "Write a function product_list(xs) that returns the product of the numbers in xs, 1 if empty.",
   "def product_list(xs):\n    p = 1\n    for x in xs:\n        p *= x\n    return p",
   "assert product_list([2, 3, 4]) == 24\nassert product_list([]) == 1")
_t("max_list", 2, "Write a function max_list(xs) that returns the largest item of the non-empty list xs.",
   "def max_list(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m",
   "assert max_list([3, 9, 2]) == 9\nassert max_list([-5]) == -5")
_t("count_even", 2, "Write a function count_even(xs) that counts the even numbers in xs.",
   "def count_even(xs):\n    n = 0\n    for x in xs:\n        if x % 2 == 0:\n            n += 1\n    return n",
   "assert count_even([1, 2, 3, 4]) == 2\nassert count_even([]) == 0")
_t("reverse_str", 1, "Write a function reverse_str(s) that returns the string s reversed.",
   "def reverse_str(s):\n    return s[::-1]", "assert reverse_str('abc') == 'cba'\nassert reverse_str('') == ''")
_t("upper", 1, "Write a function upper(s) that returns s in upper case.", "def upper(s):\n    return s.upper()",
   "assert upper('abc') == 'ABC'")
_t("first_char", 1, "Write a function first_char(s) that returns the first character of a non-empty string s.",
   "def first_char(s):\n    return s[0]", "assert first_char('xyz') == 'x'")
_t("last_char", 1, "Write a function last_char(s) that returns the last character of a non-empty string s.",
   "def last_char(s):\n    return s[-1]", "assert last_char('xyz') == 'z'")
_t("str_len", 1, "Write a function str_len(s) that returns the number of characters in s without using len.",
   "def str_len(s):\n    n = 0\n    for _ in s:\n        n += 1\n    return n", "assert str_len('hello') == 5\nassert str_len('') == 0")
_t("is_palindrome", 2, "Write a function is_palindrome(s) that returns True if s reads the same backwards.",
   "def is_palindrome(s):\n    return s == s[::-1]",
   "assert is_palindrome('level')\nassert not is_palindrome('abc')\nassert is_palindrome('')")
_t("count_vowels", 2, "Write a function count_vowels(s) that counts the letters a, e, i, o, u in lower case string s.",
   "def count_vowels(s):\n    n = 0\n    for c in s:\n        if c in 'aeiou':\n            n += 1\n    return n",
   "assert count_vowels('education') == 5\nassert count_vowels('xyz') == 0")
_t("repeat_str", 1, "Write a function repeat_str(s, n) that returns s repeated n times.",
   "def repeat_str(s, n):\n    return s * n", "assert repeat_str('ab', 3) == 'ababab'\nassert repeat_str('a', 0) == ''")
_t("fact", 2, "Write a function fact(n) that returns n factorial for n >= 0.",
   "def fact(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
   "assert fact(0) == 1\nassert fact(5) == 120")
_t("fib", 2, "Write a function fib(n) that returns the n-th Fibonacci number with fib(0) = 0 and fib(1) = 1.",
   "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
   "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55")
_t("sum_to", 1, "Write a function sum_to(n) that returns the sum of the integers from 1 to n.",
   "def sum_to(n):\n    return n * (n + 1) // 2", "assert sum_to(4) == 10\nassert sum_to(0) == 0")
_t("gcd", 2, "Write a function gcd(a, b) that returns the greatest common divisor of two positive integers.",
   "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a", "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1")
_t("is_prime", 3, "Write a function is_prime(n) that returns True if n is a prime number.",
   "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
   "assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\nassert not is_prime(15)")
_t("power", 2, "Write a function power(b, e) that returns b to the power e for e >= 0 using a loop.",
   "def power(b, e):\n    r = 1\n    for _ in range(e):\n        r *= b\n    return r", "assert power(2, 10) == 1024\nassert power(5, 0) == 1")
_t("evens", 2, "Write a function evens(xs) that returns a list of the even numbers in xs.",
   "def evens(xs):\n    return [x for x in xs if x % 2 == 0]", "assert evens([1, 2, 3, 4]) == [2, 4]\nassert evens([1]) == []")
_t("squares", 2, "Write a function squares(n) that returns the list of squares of 0 up to n-1.",
   "def squares(n):\n    return [i * i for i in range(n)]", "assert squares(4) == [0, 1, 4, 9]\nassert squares(0) == []")
_t("rev_list", 1, "Write a function rev_list(xs) that returns a new list with the items of xs reversed.",
   "def rev_list(xs):\n    return xs[::-1]", "assert rev_list([1, 2, 3]) == [3, 2, 1]")
_t("contains", 1, "Write a function contains(xs, v) that returns True if v is in the list xs.",
   "def contains(xs, v):\n    for x in xs:\n        if x == v:\n            return True\n    return False",
   "assert contains([1, 2], 2)\nassert not contains([1, 2], 3)")
_t("index_of", 2, "Write a function index_of(xs, v) that returns the first index of v in xs, or -1.",
   "def index_of(xs, v):\n    for i, x in enumerate(xs):\n        if x == v:\n            return i\n    return -1",
   "assert index_of([5, 6, 7], 6) == 1\nassert index_of([5], 9) == -1")
_t("dedupe", 3, "Write a function dedupe(xs) that removes duplicates from xs keeping the first occurrence order.",
   "def dedupe(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
   "assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]\nassert dedupe([]) == []")
_t("swap_pair", 1, "Write a function swap_pair(p) that takes a tuple (a, b) and returns (b, a).",
   "def swap_pair(p):\n    a, b = p\n    return (b, a)", "assert swap_pair((1, 2)) == (2, 1)")
_t("celsius", 1, "Write a function celsius(f) that converts Fahrenheit f to Celsius as (f - 32) * 5 / 9.",
   "def celsius(f):\n    return (f - 32) * 5 / 9", "assert celsius(32) == 0\nassert celsius(212) == 100")
_t("fizzbuzz", 3, "Write a function fizzbuzz(n) returning 'FizzBuzz' if n divisible by 15, 'Fizz' if by 3, 'Buzz' if by 5, otherwise str(n).",
   "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
   "assert fizzbuzz(15) == 'FizzBuzz'\nassert fizzbuzz(9) == 'Fizz'\nassert fizzbuzz(10) == 'Buzz'\nassert fizzbuzz(7) == '7'")
_t("count_char", 2, "Write a function count_char(s, c) that counts how many times character c occurs in s.",
   "def count_char(s, c):\n    n = 0\n    for x in s:\n        if x == c:\n            n += 1\n    return n", "assert count_char('banana', 'a') == 3\nassert count_char('abc', 'z') == 0")
_t("word_count", 2, "Write a function word_count(s) that returns the number of whitespace separated words in s.",
   "def word_count(s):\n    return len(s.split())", "assert word_count('a b c') == 3\nassert word_count('') == 0")
_t("digit_sum", 2, "Write a function digit_sum(n) that returns the sum of the decimal digits of the non-negative integer n.",
   "def digit_sum(n):\n    t = 0\n    while n > 0:\n        t += n % 10\n        n //= 10\n    return t", "assert digit_sum(123) == 6\nassert digit_sum(0) == 0")
_t("binary_str", 3, "Write a function binary_str(n) that returns n >= 0 as a binary string without prefix, '0' for zero.",
   "def binary_str(n):\n    if n == 0:\n        return '0'\n    s = ''\n    while n > 0:\n        s = str(n % 2) + s\n        n //= 2\n    return s", "assert binary_str(5) == '101'\nassert binary_str(0) == '0'\nassert binary_str(8) == '1000'")
_t("bubble_sort", 3, "Write a function bubble_sort(xs) that returns a sorted copy of xs using bubble sort.",
   "def bubble_sort(xs):\n    a = list(xs)\n    for i in range(len(a)):\n        for j in range(len(a) - 1 - i):\n            if a[j] > a[j + 1]:\n                a[j], a[j + 1] = a[j + 1], a[j]\n    return a",
   "assert bubble_sort([3, 1, 2]) == [1, 2, 3]\nassert bubble_sort([]) == []\nassert bubble_sort([2, 2, 1]) == [1, 2, 2]")
_t("second_largest", 3, "Write a function second_largest(xs) that returns the second largest distinct value in xs.",
   "def second_largest(xs):\n    vals = sorted(set(xs))\n    return vals[-2]", "assert second_largest([1, 5, 3]) == 3\nassert second_largest([4, 4, 2]) == 2")

TASKS = _T
