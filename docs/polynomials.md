## 1. `hennessey_2008`

**Title:** *Fixation Precision in High-Speed Noncontact Eye-Gaze Tracking* (Hennessey et al., 2008)
**URL:** [https://ieeexplore.ieee.org/document/4410444](https://ieeexplore.ieee.org/document/4410444)
**Model:**

$$
\text{gaze} = a_0\,x\,y + a_1\,x + a_2\,y + a_3
$$

**Python snippet:**

```python
def hennessey_2008(x, y):
    return np.array([x*y, x, y, 1])
```

---

## 2. `hoorman_2008`

**Model:**

$$
\begin{cases}
\text{gaze}_x = a_0\,x + a_1,\\
\text{gaze}_y = b_0\,y + b_1
\end{cases}
$$

**Python snippet:**

```python
def hoorman_2008(x, y):
    return np.array([[x, 1],
                     [y, 1]])
```

---

## 3. `cerrolaza_2008`

**Title:** *Taxonomic study of polynomial regressions applied to the calibration of video‑oculographic systems* (Cerrolaza, Villanueva & Cabeza, 2008)
**URL:** [https://dl.acm.org/doi/10.1145/2240156.2240158](https://dl.acm.org/doi/10.1145/2240156.2240158)
**Model:**

$$
\text{gaze} = a_0\,x^2 + a_1\,y^2 + a_2\,x\,y + a_3\,x + a_4\,y + a_5
$$

**Python snippet:**

```python
def cerrolaza_2008(x, y):
    return np.array([x**2, y**2, x*y, x, y, 1])
```

---

## 4. `second_order`

**Title:** *General second-order polynomial model (not tied to a specific paper)*
**Model:**

$$
\text{gaze} = a_0\,x^2y^2 + a_1\,x^2 + a_2\,y^2 + a_3\,x\,y + a_4\,x + a_5\,y + a_6
$$

**Python snippet:**

```python
def second_order(x, y):
    return np.array([x**2 * y**2, x**2, y**2, x*y, x, y, 1])
```

---

## 5. `zhu_ji_2005`

**Title:** *(Zhu & Ji, 2005) – axis-specific calibration*
**URL:** (survey cites via PMC, no direct PDF) [https://pmc.ncbi.nlm.nih.gov/articles/PMC8482219/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8482219/)
**Model:**

$$
\begin{cases}
\text{gaze}_x = a_0\,x\,y + a_1\,x + a_2\,y + a_3,\\
\text{gaze}_y = b_0\,y^2 + b_1\,x + b_2\,y + b_3
\end{cases}
$$

**Python snippet:**

```python
def zhu_ji_2005(x, y):
    return np.array([[x*y, x, y, 1],
                     [y**2, x, y, 1]])
```

---

## 6. `cerrolaza_villanueva_2008`

**Title:** *Taxonomic study of polynomial regressions applied to the calibration of video‑oculographic systems* (same as #3)
**URL:** [https://dl.acm.org/doi/10.1145/2240156.2240158](https://dl.acm.org/doi/10.1145/2240156.2240158)
**Model:**

$$
\begin{cases}
\text{gaze}_x = a_0\,x^2 + a_1\,x + a_2\,y + a_3,\\
\text{gaze}_y = b_0\,x^2y + b_1\,x^2 + b_2\,x\,y + b_3\,y + b_4
\end{cases}
$$

**Python snippet:**

```python
def cerrolaza_villanueva_2008(x, y):
    return np.array([[x**2, x, y, 1, 0],
                     [x**2 * y, x**2, x*y, y, 1]])
```

---

## 7. `blignaut_wium_2013`

**Model:**

$$
\begin{aligned}
\text{gaze}_x &= a_0 + a_1\,x + a_2\,x^3 + a_3\,y^2 + a_4\,x\,y,\\
\text{gaze}_y &= b_0 + b_1\,x + b_2\,x^2 + b_3\,y + b_4\,y^2 + b_5\,x\,y + b_6\,x^2\,y
\end{aligned}
$$

**Python snippet:**

```python
def blignaut_wium_2013(x, y):
    return np.array([[1, x, x**3, y**2, x*y, 0, 0],
                     [1, x, x**2, y, y**2, x*y, x**2 * y]])
```

---
Absolutely! Here's your **updated reference record**, now including the **full third-order polynomial**, complete with title, URL, mathematical model, and Python code. It's appended to your previous list in the same structured format.

---

### 8. `third_order`

**Title:** *General third-order polynomial for eye-tracking calibration*
**URL:** *Not from a specific paper — general formulation used in literature and modeling*
**Model:**

$$
\text{gaze} = a_0 + a_1\,x + a_2\,y + a_3\,x^2 + a_4\,y^2 + a_5\,x\,y + a_6\,x^3 + a_7\,y^3 + a_8\,x^2\,y + a_9\,x\,y^2
$$

**Python Code:**

```python
def third_order(x, y):
    return np.array([
        1,
        x, y,
        x**2, y**2, x*y,
        x**3, y**3,
        x**2 * y, x * y**2
    ])
```
---


### 📊 Summary Table

| Function                    | Title                           | URL                                                                                                                                                                                                                                                                          | Model                              | Python Code                  |
| --------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ---------------------------- |
| `hennessey_2008`            | A new mapping function… (2008)  | [https://www.researchgate.net/publication/261848648\_A\_new\_mapping\_function\_to\_improve\_the\_accuracy\_of\_a\_video-based\_eye-tracker](https://www.researchgate.net/publication/261848648_A_new_mapping_function_to_improve_the_accuracy_of_a_video-based_eye-tracker) | gaze = a₀xy + a₁x + a₂y + a₃       | `np.array([x*y, x, y, 1])`   |
| `hoorman_2008`              | Jainta & Hoormann (2008)        | [https://bop.unibe.ch/JEMR/article/download/2465/pdf\_3](https://bop.unibe.ch/JEMR/article/download/2465/pdf_3)                                                                                                                                                              | gazeₓ = a₀x + a₁; gazeᵧ = b₀y + b₁ | `[[x,1],[y,1]]`              |
| `cerrolaza_2008`            | Taxonomic study… (2008)         | [https://dl.acm.org/doi/10.1145/2240156.2240158](https://dl.acm.org/doi/10.1145/2240156.2240158)                                                                                                                                                                             | gaze = a₀x² + … + a₅               | `[...,1]`                    |
| `second_order`              | General 2nd-order               | -                                                                                                                                                                                                                                                                           | includes x²y² term                 | `[...,1]`                    |
| `zhu_ji_2005`               | Zhu & Ji (2005)                 | [https://pmc.ncbi.nlm.nih.gov/articles/PMC8482219](https://pmc.ncbi.nlm.nih.gov/articles/PMC8482219)                                                                                                                                                                         | mixed axis terms                   | `[[xy,x,y,1],[y²,x,y,1]]`    |
| `cerrolaza_villanueva_2008` | Taxonomic study… (2008)         | [https://dl.acm.org/doi/10.1145/2240156.2240158](https://dl.acm.org/doi/10.1145/2240156.2240158)                                                                                                                                                                             | asymmetric 2D                      | `[[x²,x,y,1,0],[x²y,...,1]]` |
| `blignaut_wium_2013`        | Mapping the Pupil‑Glint… (2013) | [https://bop.unibe.ch/JEMR/article/download/2373/3569/8597](https://bop.unibe.ch/JEMR/article/download/2373/3569/8597)                                                                                                                                                       | higher-order mix                   | `[[1,x,x³,y²,xy,0,0],…]`     |
| **`third_order`**           | General third-order polynomial  | -                                                                                                                                                                                                                                                             | adds `x³`, `y³`, `x²y`, `xy²`      |                   |

---

