Sure! Here's the epsilon-delta definition of a limit using proper math delimiters for markdown:

### Inline math:

The limit of a function $f(x)$ as $x$ approaches $c$ is $L$ if for every $\epsilon > 0$, there exists a $\delta > 0$ such that if $0 < |x - c| < \delta$, then $|f(x) - L| < \epsilon$.

### Block math:

In block form, you can write it as:

$$
\lim_{x \to c} f(x) = L \quad \text{if for every} \, \epsilon > 0, \, \text{there exists a} \, \delta > 0 \, \text{such that if} \, 0 < |x - c| < \delta, \, \text{then} \, |f(x) - L| < \epsilon.
$$

Make sure to use the appropriate extension (like Markdown Preview Enhanced) in VSCode to properly render these formulas!

Inline math: $|f(x) - L| < \epsilon$

Block math:

$$
\lim_{x \to c} f(x) = L \quad \text{if for every } \epsilon > 0, \text{ there exists a } \delta > 0 \text{ such that if } 0 < |x - c| < \delta, \text{ then } |f(x) - L| < \epsilon.
$$


# Epsilon–Delta Definition of a Limit

[Formal definition of limits Part 4: using the definition (video) | Khan Academy](https://www.khanacademy.org/math/calculus-1/cs1-limits-and-continuity/cs1-formal-definition-of-limits-epsilon-delta/v/proving-a-limit-using-epsilon-delta-definition)


Let $f(x)$ be a function defined on an open interval around $a$ (except possibly at $a$).
We say that:

$$
\lim_{x \to a} f(x) = L
$$

if for every $\varepsilon > 0$, there exists a $\delta > 0$ such that:

$$
0 < |x - a| < \delta \Rightarrow |f(x) - L| < \varepsilon
$$

---

# Proof that $\lim_{x \to 2} (3x + 1) = 7$

We want to show that for every $\varepsilon > 0$, there exists a $\delta > 0$ such that:

$$
0 < |x - 2| < \delta \Rightarrow |(3x + 1) - 7| < \varepsilon
$$

## Step 1: Simplify the absolute value expression

We simplify:

$$
|(3x + 1) - 7| = |3x - 6| = 3|x - 2|
$$

We want:

$$
3|x - 2| < \varepsilon \Rightarrow |x - 2| < \frac{\varepsilon}{3}
$$

## Step 2: Choose $\delta$

Let $\delta = \frac{\varepsilon}{3}$. Then if $0 < |x - 2| < \delta$, we have:

$$
3|x - 2| < 3 \cdot \frac{\varepsilon}{3} = \varepsilon
$$

Therefore:

$$
|(3x + 1) - 7| < \varepsilon
$$

---

✅ **Conclusion**:
This proves that $\lim_{x \to 2} (3x + 1) = 7$ using the epsilon–delta definition.
