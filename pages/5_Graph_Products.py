import streamlit as st
import pandas as pd
from fractions import Fraction
from functions import (
    make_all_upper_linear_conjectures,
    make_all_lower_linear_conjectures,
    dalmatian,
    filter_conjectures,
    invariants,
    booleans,
)
import numpy as np
import time

# Move set_page_config to the top to avoid potential errors later
# st.set_page_config(page_title="Conjecture Generator")

DATA_FILE = "training-data/full-graph-product-data.csv"

TEX_MAP = {
    "is_connected": r"If $G \text{ and } H \text{ are nontrivial onnected graphs, then, }$",
    "<=": r"\leq",
    ">=": r"\geq",

    "cartesian_product_domination_number": r"\gamma(G \mathbin{\square} H)",
    "lexicographic_product_domination_number": r"\gamma(G \circ H)",
    "tensor_product_domination_number": r"\gamma(G \times H)",
    "strong_product_domination_number": r"\gamma(G \boxtimes H)",

    "domination_number * domination_number": r"\gamma(G) \cdot \gamma(H)",
    "domination_number + domination_number": r"(\gamma(G) + \gamma(H))",
    "domination_number * total_domination_number": r"\gamma(G) \cdot \gamma_t(H)",
    "domination_number + total_domination_number": r"(\gamma(G) + \gamma_t(H))",
    "domination_number * independence_number": r"\gamma(G) \cdot \alpha(H)",
    "domination_number + independence_number": r"(\gamma(G) + \alpha(H))",
    "domination_number * independent_domination_number": r"\gamma(G) \cdot i(H)",
    "domination_number + independent_domination_number": r"(\gamma(G) + i(H))",
    "domination_number * chromatic_number": r"\gamma(G) \cdot \chi(H)",
    "domination_number + chromatic_number": r"(\gamma(G) + \chi(H))",
    "domination_number * matching_number": r"\gamma(G) \cdot \mu(H)",
    "domination_number + matching_number": r"(\gamma(G) + \mu(H))",
    "domination_number * edge_domination_number": r"\gamma(G) \cdot \gamma_e(H)",
    "domination_number + edge_domination_number": r"(\gamma(G) + \gamma_e(H))",
    "domination_number * clique_number": r"\gamma(G) \cdot \omega(H)",
    "domination_number + clique_number": r"(\gamma(G) + \omega(H))",
    "domination_number * residue": r"\gamma(G) \cdot R(H)",
    "domination_number + residue": r"(\gamma(G) + R(H))",
    "domination_number * annihilation_number": r"\gamma(G) \cdot a(H)",
    "domination_number + annihilation_number": r"(\gamma(G) + a(H))",
    "domination_number * slater": r"\gamma(G) \cdot sl(H)",
    "domination_number + slater": r"(\gamma(G) + sl(H))",
    "domination_number * vertex_cover_number": r"\gamma(G) \cdot \beta(H)",
    "domination_number + vertex_cover_number": r"(\gamma(G) + \beta(H))",
    "domination_number * roman_domination_number": r"\gamma(G) \cdot \gamma_{R}(H)",
    "domination_number + roman_domination_number": r"(\gamma(G) + \gamma_{R}(H))",
    "domination_number * double_roman_domination_number": r"\gamma(G) \cdot \gamma_{dR}(H)",
    "domination_number + double_roman_domination_number": r"(\gamma(G) + \gamma_{dR}(H))",
    "domination_number * two_rainbow_domination_number": r"\gamma(G) \cdot \gamma_{r2}(H)",
    "domination_number + two_rainbow_domination_number": r"(\gamma(G) + \gamma_{r2}(H))",
    "domination_number * three_rainbow_domination_number": r"\gamma(G) \cdot \gamma_{r3}(H)",
    "domination_number + three_rainbow_domination_number": r"(\gamma(G) + \gamma_{r3}(H))",
    "domination_number * restrained_domination_number": r"\gamma(G) \cdot \gamma_{r}(H)",
    "domination_number + restrained_domination_number": r"(\gamma(G) + \gamma_{r}(H))",
    "domination_number * outer_connected_domination_number": r"\gamma(G) \cdot \tilde{\gamma}_{c}(H)",
    "domination_number + outer_connected_domination_number": r"(\gamma(G) + \tilde{\gamma}_{c}(H))",

    "cartesian_product_total_domination_number": r"\gamma_t(G \mathbin{\square} H)",
    "lexicographic_product_total_domination_number": r"\gamma_t(G \circ H)",
    "tensor_product_total_domination_number": r"\gamma_t(G \times H)",
    "strong_product_total_domination_number": r"\gamma_t(G \boxtimes H)",

    "total_domination_number * domination_number": r"\gamma_t(G) \cdot \gamma(H)",
    "total_domination_number + domination_number": r"(\gamma_t(G) + \gamma(H))",
    "total_domination_number * total_domination_number": r"\gamma_t(G) \cdot \gamma_t(H)",
    "total_domination_number + total_domination_number": r"(\gamma_t(G) + \gamma_t(H))",
    "total_domination_number * independence_number": r"\gamma_t(G) \cdot \alpha(H)",
    "total_domination_number + independence_number": r"(\gamma_t(G) + \alpha(H))",
    "total_domination_number * independent_domination_number": r"\gamma_t(G) \cdot i(H)",
    "total_domination_number + independent_domination_number": r"(\gamma_t(G) + i(H))",
    "total_domination_number * chromatic_number": r"\gamma_t(G) \cdot \chi(H)",
    "total_domination_number + chromatic_number": r"(\gamma_t(G) + \chi(H))",
    "total_domination_number * matching_number": r"\gamma_t(G) \cdot \mu(H)",
    "total_domination_number + matching_number": r"(\gamma_t(G) + \mu(H))",
    "total_domination_number * edge_domination_number": r"\gamma_t(G) \cdot \gamma_e(H)",
    "total_domination_number + edge_domination_number": r"(\gamma_t(G) + \gamma_e(H))",
    "total_domination_number * clique_number": r"\gamma_t(G) \cdot \omega(H)",
    "total_domination_number + clique_number": r"(\gamma_t(G) + \omega(H))",
    "total_domination_number * residue": r"\gamma_t(G) \cdot R(H)",
    "total_domination_number + residue": r"(\gamma_t(G) + R(H))",
    "total_domination_number * annihilation_number": r"\gamma_t(G) \cdot a(H)",
    "total_domination_number + annihilation_number": r"(\gamma_t(G) + a(H))",
    "total_domination_number * slater": r"\gamma_t(G) \cdot sl(H)",
    "total_domination_number + slater": r"(\gamma_t(G) + sl(H))",
    "total_domination_number * vertex_cover_number": r"\gamma_t(G) \cdot \beta(H)",
    "total_domination_number + vertex_cover_number": r"(\gamma_t(G) + \beta(H))",
    "total_domination_number * roman_domination_number": r"\gamma_t(G) \cdot \gamma_{R}(H)",
    "total_domination_number + roman_domination_number": r"(\gamma_t(G) + \gamma_{R}(H))",
    "total_domination_number * double_roman_domination_number": r"\gamma_t(G) \cdot \gamma_{dR}(H)",
    "total_domination_number + double_roman_domination_number": r"(\gamma_t(G) + \gamma_{dR}(H))",
    "total_domination_number * two_rainbow_domination_number": r"\gamma_t(G) \cdot \gamma_{r2}(H)",
    "total_domination_number + two_rainbow_domination_number": r"(\gamma_t(G) + \gamma_{r2}(H))",
    "total_domination_number * three_rainbow_domination_number": r"\gamma_t(G) \cdot \gamma_{r3}(H)",
    "total_domination_number + three_rainbow_domination_number": r"(\gamma_t(G) + \gamma_{r3}(H))",
    "total_domination_number * restrained_domination_number": r"\gamma_t(G) \cdot \gamma_{r}(H)",
    "total_domination_number + restrained_domination_number": r"(\gamma_t(G) + \gamma_{r}(H))",
    "total_domination_number * outer_connected_domination_number": r"\gamma_t(G) \cdot \tilde{\gamma}_{c}(H)",
    "total_domination_number + outer_connected_domination_number": r"(\gamma_t(G) + \tilde{\gamma}_{c}(H))",

    "cartesian_product_independence_number": r"\alpha(G \mathbin{\square} H)",
    "lexicographic_product_independence_number": r"\alpha(G \circ H)",
    "tensor_product_independence_number": r"\alpha(G \times H)",
    "strong_product_independence_number": r"\alpha(G \boxtimes H)",

    "independence_number * domination_number": r"\alpha(G) \cdot \gamma(H)",
    "independence_number + domination_number": r"(\alpha(G) + \gamma(H))",
    "independence_number * total_domination_number": r"\alpha(G) \cdot \gamma_t(H)",
    "independence_number + total_domination_number": r"(\alpha(G) + \gamma_t(H))",
    "independence_number * independence_number": r"\alpha(G) \cdot \alpha(H)",
    "independence_number + independence_number": r"(\alpha(G) + \alpha(H))",
    "independence_number * independent_domination_number": r"\alpha(G) \cdot i(H)",
    "independence_number + independent_domination_number": r"(\alpha(G) + i(H))",
    "independence_number * chromatic_number": r"\alpha(G) \cdot \chi(H)",
    "independence_number + chromatic_number": r"(\alpha(G) + \chi(H))",
    "independence_number * matching_number": r"\alpha(G) \cdot \mu(H)",
    "independence_number + matching_number": r"(\alpha(G) + \mu(H))",
    "independence_number * edge_domination_number": r"\alpha(G) \cdot \gamma_e(H)",
    "independence_number + edge_domination_number": r"(\alpha(G) + \gamma_e(H))",
    "independence_number * clique_number": r"\alpha(G) \cdot \omega(H)",
    "independence_number + clique_number": r"(\alpha(G) + \omega(H))",
    "independence_number * residue": r"\alpha(G) \cdot R(H)",
    "independence_number + residue": r"(\alpha(G) + R(H))",
    "independence_number * annihilation_number": r"\alpha(G) \cdot a(H)",
    "independence_number + annihilation_number": r"(\alpha(G) + a(H))",
    "independence_number * slater": r"\alpha(G) \cdot sl(H)",
    "independence_number + slater": r"(\alpha(G) + sl(H))",
    "independence_number * vertex_cover_number": r"\alpha(G) \cdot \beta(H)",
    "independence_number + vertex_cover_number": r"(\alpha(G) + \beta(H))",
    "independence_number * roman_domination_number": r"\alpha(G) \cdot \gamma_{R}(H)",
    "independence_number + roman_domination_number": r"(\alpha(G) + \gamma_{R}(H))",
    "independence_number * double_roman_domination_number": r"\alpha(G) \cdot \gamma_{dR}(H)",
    "independence_number + double_roman_domination_number": r"(\alpha(G) + \gamma_{dR}(H))",
    "independence_number * two_rainbow_domination_number": r"\alpha(G) \cdot \gamma_{r2}(H)",
    "independence_number + two_rainbow_domination_number": r"(\alpha(G) + \gamma_{r2}(H))",
    "independence_number * three_rainbow_domination_number": r"\alpha(G) \cdot \gamma_{r3}(H)",
    "independence_number + three_rainbow_domination_number": r"(\alpha(G) + \gamma_{r3}(H))",
    "independence_number * restrained_domination_number": r"\alpha(G) \cdot \gamma_{r}(H)",
    "independence_number + restrained_domination_number": r"(\alpha(G) + \gamma_{r}(H))",
    "independence_number * outer_connected_domination_number": r"\alpha(G) \cdot \tilde{\gamma}_{c}(H)",
    "independence_number + outer_connected_domination_number": r"(\alpha(G) + \tilde{\gamma}_{c}(H))",

    "cartesian_product_independent_domination_number": r"i(G \mathbin{\square} H)",
    "lexicographic_product_independent_domination_number": r"i(G \circ H)",
    "tensor_product_independent_domination_number": r"i(G \times H)",
    "strong_product_independent_domination_number": r"i(G \boxtimes H)",

    "independent_domination_number * domination_number": r"i(G) \cdot \gamma(H)",
    "independent_domination_number + domination_number": r"(i(G) + \gamma(H))",
    "independent_domination_number * total_domination_number": r"i(G) \cdot \gamma_t(H)",
    "independent_domination_number + total_domination_number": r"(i(G) + \gamma_t(H))",
    "independent_domination_number * independence_number": r"i(G) \cdot \alpha(H)",
    "independent_domination_number + independence_number": r"(i(G) + \alpha(H))",
    "independent_domination_number * independent_domination_number": r"i(G) \cdot i(H)",
    "independent_domination_number + independent_domination_number": r"(i(G) + i(H))",
    "independent_domination_number * chromatic_number": r"i(G) \cdot \chi(H)",
    "independent_domination_number + chromatic_number": r"(i(G) + \chi(H))",
    "independent_domination_number * matching_number": r"i(G) \cdot \mu(H)",
    "independent_domination_number + matching_number": r"(i(G) + \mu(H))",
    "independent_domination_number * edge_domination_number": r"i(G) \cdot \gamma_e(H)",
    "independent_domination_number + edge_domination_number": r"(i(G) + \gamma_e(H))",
    "independent_domination_number * clique_number": r"i(G) \cdot \omega(H)",
    "independent_domination_number + clique_number": r"(i(G) + \omega(H))",
    "independent_domination_number * residue": r"i(G) \cdot R(H)",
    "independent_domination_number + residue": r"(i(G) + R(H))",
    "independent_domination_number * annihilation_number": r"i(G) \cdot a(H)",
    "independent_domination_number + annihilation_number": r"(i(G) + a(H))",
    "independent_domination_number * slater": r"i(G) \cdot sl(H)",
    "independent_domination_number + slater": r"(i(G) + sl(H))",
    "independent_domination_number * vertex_cover_number": r"i(G) \cdot \beta(H)",
    "independent_domination_number + vertex_cover_number": r"(i(G) + \beta(H))",
    "independent_domination_number * roman_domination_number": r"i(G) \cdot \gamma_{R}(H)",
    "independent_domination_number + roman_domination_number": r"(i(G) + \gamma_{R}(H))",
    "independent_domination_number * double_roman_domination_number": r"i(G) \cdot \gamma_{dR}(H)",
    "independent_domination_number + double_roman_domination_number": r"(i(G) + \gamma_{dR}(H))",
    "independent_domination_number * two_rainbow_domination_number": r"i(G) \cdot \gamma_{r2}(H)",
    "independent_domination_number + two_rainbow_domination_number": r"(i(G) + \gamma_{r2}(H))",
    "independent_domination_number * three_rainbow_domination_number": r"i(G) \cdot \gamma_{r3}(H)",
    "independent_domination_number + three_rainbow_domination_number": r"(i(G) + \gamma_{r3}(H))",
    "independent_domination_number * restrained_domination_number": r"i(G) \cdot \gamma_{r}(H)",
    "independent_domination_number + restrained_domination_number": r"(i(G) + \gamma_{r}(H))",
    "independent_domination_number * outer_connected_domination_number": r"i(G) \cdot \tilde{\gamma}_{c}(H)",
    "independent_domination_number + outer_connected_domination_number": r"(i(G) + \tilde{\gamma}_{c}(H))",

    "cartesian_product_chromatic_number": r"\chi(G \mathbin{\square} H)",
    "lexicographic_product_chromatic_number": r"\chi(G \circ H)",
    "tensor_product_chromatic_number": r"\chi(G \times H)",
    "strong_product_chromatic_number": r"\chi(G \boxtimes H)",

    "chromatic_number * domination_number": r"\chi(G) \cdot \gamma(H)",
    "chromatic_number + domination_number": r"(\chi(G) + \gamma(H))",
    "chromatic_number * total_domination_number": r"\chi(G) \cdot \gamma_t(H)",
    "chromatic_number + total_domination_number": r"(\chi(G) + \gamma_t(H))",
    "chromatic_number * independence_number": r"\chi(G) \cdot \alpha(H)",
    "chromatic_number + independence_number": r"(\chi(G) + \alpha(H))",
    "chromatic_number * chromatic_number": r"\chi(G) \cdot \chi(H)",
    "chromatic_number + chromatic_number": r"(\chi(G) + \chi(H))",
    "chromatic_number * matching_number": r"\chi(G) \cdot \mu(H)",
    "chromatic_number + matching_number": r"(\chi(G) + \mu(H))",
    "chromatic_number * edge_domination_number": r"\chi(G) \cdot \gamma_e(H)",
    "chromatic_number + edge_domination_number": r"(\chi(G) + \gamma_e(H))",
    "chromatic_number * clique_number": r"\chi(G) \cdot \omega(H)",
    "chromatic_number + clique_number": r"(\chi(G) + \omega(H))",
    "chromatic_number * residue": r"\chi(G) \cdot R(H)",
    "chromatic_number + residue": r"(\chi(G) + R(H))",
    "chromatic_number * annihilation_number": r"\chi(G) \cdot a(H)",
    "chromatic_number + annihilation_number": r"(\chi(G) + a(H))",
    "chromatic_number * slater": r"\chi(G) \cdot sl(H)",
    "chromatic_number + slater": r"(\chi(G) + sl(H))",
    "chromatic_number * vertex_cover_number": r"\chi(G) \cdot \beta(H)",
    "chromatic_number + vertex_cover_number": r"(\chi(G) + \beta(H))",
    "chromatic_number * roman_domination_number": r"\chi(G) \cdot \gamma_{R}(H)",
    "chromatic_number + roman_domination_number": r"(\chi(G) + \gamma_{R}(H))",
    "chromatic_number * double_roman_domination_number": r"\chi(G) \cdot \gamma_{dR}(H)",
    "chromatic_number + double_roman_domination_number": r"(\chi(G) + \gamma_{dR}(H))",
    "chromatic_number * two_rainbow_domination_number": r"\chi(G) \cdot \gamma_{r2}(H)",
    "chromatic_number + two_rainbow_domination_number": r"(\chi(G) + \gamma_{r2}(H))",
    "chromatic_number * three_rainbow_domination_number": r"\chi(G) \cdot \gamma_{r3}(H)",
    "chromatic_number + three_rainbow_domination_number": r"(\chi(G) + \gamma_{r3}(H))",
    "chromatic_number * restrained_domination_number": r"\chi(G) \cdot \gamma_{r}(H)",
    "chromatic_number + restrained_domination_number": r"(\chi(G) + \gamma_{r}(H))",
    "chromatic_number * outer_connected_domination_number": r"\chi(G) \cdot \tilde{\gamma}_{c}(H)",
    "chromatic_number + outer_connected_domination_number": r"(\chi(G) + \tilde{\gamma}_{c}(H))",

    "cartesian_product_matching_number": r"\mu(G \mathbin{\square} H)",
    "lexicographic_product_matching_number": r"\mu(G \circ H)",
    "tensor_product_matching_number": r"\mu(G \times H)",
    "strong_product_matching_number": r"\mu(G \boxtimes H)",

    "matching_number * domination_number": r"\mu(G) \cdot \gamma(H)",
    "matching_number + domination_number": r"(\mu(G) + \gamma(H))",
    "matching_number * total_domination_number": r"\mu(G) \cdot \gamma_t(H)",
    "matching_number + total_domination_number": r"(\mu(G) + \gamma_t(H))",
    "matching_number * independence_number": r"\mu(G) \cdot \alpha(H)",
    "matching_number + independence_number": r"(\mu(G) + \alpha(H))",
    "matching_number * independent_domination_number": r"\mu(G) \cdot i(H)",
    "matching_number + independent_domination_number": r"(\mu(G) + i(H))",
    "matching_number * chromatic_number": r"\mu(G) \cdot \chi(H)",
    "matching_number + chromatic_number": r"(\mu(G) + \chi(H))",
    "matching_number * matching_number": r"\mu(G) \cdot \mu(H)",
    "matching_number + matching_number": r"(\mu(G) + \mu(H))",
    "matching_number * edge_domination_number": r"\mu(G) \cdot \gamma_e(H)",
    "matching_number + edge_domination_number": r"(\mu(G) + \gamma_e(H))",
    "matching_number * clique_number": r"\mu(G) \cdot \omega(H)",
    "matching_number + clique_number": r"(\mu(G) + \omega(H))",
    "matching_number * residue": r"\mu(G) \cdot R(H)",
    "matching_number + residue": r"(\mu(G) + R(H))",
    "matching_number * annihilation_number": r"\mu(G) \cdot a(H)",
    "matching_number + annihilation_number": r"(\mu(G) + a(H))",
    "matching_number * slater": r"\mu(G) \cdot sl(H)",
    "matching_number + slater": r"(\mu(G) + sl(H))",
    "matching_number * vertex_cover_number": r"\mu(G) \cdot \beta(H)",
    "matching_number + vertex_cover_number": r"(\mu(G) + \beta(H))",
    "matching_number * roman_domination_number": r"\mu(G) \cdot \gamma_R(H)",
    "matching_number + roman_domination_number": r"(\mu(G) + \gamma_R(H))",
    "matching_number * double_roman_domination_number": r"\mu(G) \cdot \gamma_{dR}(H)",
    "matching_number + double_roman_domination_number": r"(\mu(G) + \gamma_{dR}(H))",
    "matching_number * two_rainbow_domination_number": r"\mu(G) \cdot \gamma_{r2}(H)",
    "matching_number + two_rainbow_domination_number": r"(\mu(G) + \gamma_{r2}(H))",
    "matching_number * three_rainbow_domination_number": r"\mu(G) \cdot \gamma_{r3}(H)",
    "matching_number + three_rainbow_domination_number": r"(\mu(G) + \gamma_{r3}(H))",
    "matching_number * restrained_domination_number": r"\mu(G) \cdot \gamma_r(H)",
    "matching_number + restrained_domination_number": r"(\mu(G) + \gamma_r(H))",
    "matching_number * outer_connected_domination_number": r"\mu(G) \cdot \tilde{\gamma}_c(H)",
    "matching_number + outer_connected_domination_number": r"(\mu(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_edge_domination_number": r"\gamma_e(G \mathbin{\square} H)",
    "lexicographic_product_edge_domination_number": r"\gamma_e(G \circ H)",
    "tensor_product_edge_domination_number": r"\gamma_e(G \times H)",
    "strong_product_edge_domination_number": r"\gamma_e(G \boxtimes H)",

    "edge_domination_number * domination_number": r"\gamma_e(G) \cdot \gamma(H)",
    "edge_domination_number + domination_number": r"(\gamma_e(G) + \gamma(H))",
    "edge_domination_number * total_domination_number": r"\gamma_e(G) \cdot \gamma_t(H)",
    "edge_domination_number + total_domination_number": r"(\gamma_e(G) + \gamma_t(H))",
    "edge_domination_number * independence_number": r"\gamma_e(G) \cdot \alpha(H)",
    "edge_domination_number + independence_number": r"(\gamma_e(G) + \alpha(H))",
    "edge_domination_number * matching_number": r"\gamma_e(G) \cdot \mu(H)",
    "edge_domination_number + matching_number": r"(\gamma_e(G) + \mu(H))",
    "edge_domination_number * edge_domination_number": r"\gamma_e(G) \cdot \gamma_e(H)",
    "edge_domination_number + edge_domination_number": r"(\gamma_e(G) + \gamma_e(H))",
    "edge_domination_number * vertex_cover_number": r"\gamma_e(G) \cdot \beta(H)",
    "edge_domination_number + vertex_cover_number": r"(\gamma_e(G) + \beta(H))",
    "edge_domination_number * clique_number": r"\gamma_e(G) \cdot \omega(H)",
    "edge_domination_number + clique_number": r"(\gamma_e(G) + \omega(H))",
    "edge_domination_number * chromatic_number": r"\gamma_e(G) \cdot \chi(H)",
    "edge_domination_number + chromatic_number": r"(\gamma_e(G) + \chi(H))",
    "edge_domination_number * residue": r"\gamma_e(G) \cdot R(H)",
    "edge_domination_number + residue": r"(\gamma_e(G) + R(H))",
    "edge_domination_number * annihilation_number": r"\gamma_e(G) \cdot a(H)",
    "edge_domination_number + annihilation_number": r"(\gamma_e(G) + a(H))",
    "edge_domination_number * slater": r"\gamma_e(G) \cdot sl(H)",
    "edge_domination_number + slater": r"(\gamma_e(G) + sl(H))",
    "edge_domination_number * roman_domination_number": r"\gamma_e(G) \cdot \gamma_R(H)",
    "edge_domination_number + roman_domination_number": r"(\gamma_e(G) + \gamma_R(H))",
    "edge_domination_number * double_roman_domination_number": r"\gamma_e(G) \cdot \gamma_{dR}(H)",
    "edge_domination_number + double_roman_domination_number": r"(\gamma_e(G) + \gamma_{dR}(H))",
    "edge_domination_number * two_rainbow_domination_number": r"\gamma_e(G) \cdot \gamma_{r2}(H)",
    "edge_domination_number + two_rainbow_domination_number": r"(\gamma_e(G) + \gamma_{r2}(H))",
    "edge_domination_number * three_rainbow_domination_number": r"\gamma_e(G) \cdot \gamma_{r3}(H)",
    "edge_domination_number + three_rainbow_domination_number": r"(\gamma_e(G) + \gamma_{r3}(H))",
    "edge_domination_number * restrained_domination_number": r"\gamma_e(G) \cdot \gamma_r(H)",
    "edge_domination_number + restrained_domination_number": r"(\gamma_e(G) + \gamma_r(H))",
    "edge_domination_number * outer_connected_domination_number": r"\gamma_e(G) \cdot \tilde{\gamma}_c(H)",
    "edge_domination_number + outer_connected_domination_number": r"(\gamma_e(G) + \tilde{\gamma}_c(H))",


    "cartesian_product_vertex_cover_number": r"\beta(G \mathbin{\square} H)",
    "lexicographic_product_vertex_cover_number": r"\beta(G \circ H)",
    "tensor_product_vertex_cover_number": r"\beta(G \times H)",
    "strong_product_vertex_cover_number": r"\beta(G \boxtimes H)",

    "vertex_cover_number * domination_number": r"\beta(G) \cdot \gamma(H)",
    "vertex_cover_number + domination_number": r"(\beta(G) + \gamma(H))",
    "vertex_cover_number * total_domination_number": r"\beta(G) \cdot \gamma_t(H)",
    "vertex_cover_number + total_domination_number": r"(\beta(G) + \gamma_t(H))",
    "vertex_cover_number * independence_number": r"\beta(G) \cdot \alpha(H)",
    "vertex_cover_number + independence_number": r"(\beta(G) + \alpha(H))",
    "vertex_cover_number * matching_number": r"\beta(G) \cdot \mu(H)",
    "vertex_cover_number + matching_number": r"(\beta(G) + \mu(H))",
    "vertex_cover_number * edge_domination_number": r"\beta(G) \cdot \gamma_e(H)",
    "vertex_cover_number + edge_domination_number": r"(\beta(G) + \gamma_e(H))",
    "vertex_cover_number * vertex_cover_number": r"\beta(G) \cdot \beta(H)",
    "vertex_cover_number + vertex_cover_number": r"(\beta(G) + \beta(H))",
    "vertex_cover_number * clique_number": r"\beta(G) \cdot \omega(H)",
    "vertex_cover_number + clique_number": r"(\beta(G) + \omega(H))",
    "vertex_cover_number * chromatic_number": r"\beta(G) \cdot \chi(H)",
    "vertex_cover_number + chromatic_number": r"(\beta(G) + \chi(H))",
    "vertex_cover_number * residue": r"\beta(G) \cdot R(H)",
    "vertex_cover_number + residue": r"(\beta(G) + R(H))",
    "vertex_cover_number * annihilation_number": r"\beta(G) \cdot a(H)",
    "vertex_cover_number + annihilation_number": r"(\beta(G) + a(H))",
    "vertex_cover_number * slater": r"\beta(G) \cdot sl(H)",
    "vertex_cover_number + slater": r"(\beta(G) + sl(H))",
    "vertex_cover_number * roman_domination_number": r"\beta(G) \cdot \gamma_R(H)",
    "vertex_cover_number + roman_domination_number": r"(\beta(G) + \gamma_R(H))",
    "vertex_cover_number * double_roman_domination_number": r"\beta(G) \cdot \gamma_{dR}(H)",
    "vertex_cover_number + double_roman_domination_number": r"(\beta(G) + \gamma_{dR}(H))",
    "vertex_cover_number * two_rainbow_domination_number": r"\beta(G) \cdot \gamma_{r2}(H)",
    "vertex_cover_number + two_rainbow_domination_number": r"(\beta(G) + \gamma_{r2}(H))",
    "vertex_cover_number * three_rainbow_domination_number": r"\beta(G) \cdot \gamma_{r3}(H)",
    "vertex_cover_number + three_rainbow_domination_number": r"(\beta(G) + \gamma_{r3}(H))",
    "vertex_cover_number * restrained_domination_number": r"\beta(G) \cdot \gamma_r(H)",
    "vertex_cover_number + restrained_domination_number": r"(\beta(G) + \gamma_r(H))",
    "vertex_cover_number * outer_connected_domination_number": r"\beta(G) \cdot \tilde{\gamma}_c(H)",
    "vertex_cover_number + outer_connected_domination_number": r"(\beta(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_clique_number": r"\omega(G \mathbin{\square} H)",
    "lexicographic_product_clique_number": r"\omega(G \circ H)",
    "tensor_product_clique_number": r"\omega(G \times H)",
    "strong_product_clique_number": r"\omega(G \boxtimes H)",

    "clique_number * domination_number": r"\omega(G) \cdot \gamma(H)",
    "clique_number + domination_number": r"(\omega(G) + \gamma(H))",
    "clique_number * total_domination_number": r"\omega(G) \cdot \gamma_t(H)",
    "clique_number + total_domination_number": r"(\omega(G) + \gamma_t(H))",
    "clique_number * independence_number": r"\omega(G) \cdot \alpha(H)",
    "clique_number + independence_number": r"(\omega(G) + \alpha(H))",
    "clique_number * matching_number": r"\omega(G) \cdot \mu(H)",
    "clique_number + matching_number": r"(\omega(G) + \mu(H))",
    "clique_number * edge_domination_number": r"\omega(G) \cdot \gamma_e(H)",
    "clique_number + edge_domination_number": r"(\omega(G) + \gamma_e(H))",
    "clique_number * vertex_cover_number": r"\omega(G) \cdot \beta(H)",
    "clique_number + vertex_cover_number": r"(\omega(G) + \beta(H))",
    "clique_number * clique_number": r"\omega(G) \cdot \omega(H)",
    "clique_number + clique_number": r"(\omega(G) + \omega(H))",
    "clique_number * chromatic_number": r"\omega(G) \cdot \chi(H)",
    "clique_number + chromatic_number": r"(\omega(G) + \chi(H))",
    "clique_number * residue": r"\omega(G) \cdot R(H)",
    "clique_number + residue": r"(\omega(G) + R(H))",
    "clique_number * annihilation_number": r"\omega(G) \cdot a(H)",
    "clique_number + annihilation_number": r"(\omega(G) + a(H))",
    "clique_number * slater": r"\omega(G) \cdot sl(H)",
    "clique_number + slater": r"(\omega(G) + sl(H))",
    "clique_number * roman_domination_number": r"\omega(G) \cdot \gamma_R(H)",
    "clique_number + roman_domination_number": r"(\omega(G) + \gamma_R(H))",
    "clique_number * double_roman_domination_number": r"\omega(G) \cdot \gamma_{dR}(H)",
    "clique_number + double_roman_domination_number": r"(\omega(G) + \gamma_{dR}(H))",
    "clique_number * two_rainbow_domination_number": r"\omega(G) \cdot \gamma_{r2}(H)",
    "clique_number + two_rainbow_domination_number": r"(\omega(G) + \gamma_{r2}(H))",
    "clique_number * three_rainbow_domination_number": r"\omega(G) \cdot \gamma_{r3}(H)",
    "clique_number + three_rainbow_domination_number": r"(\omega(G) + \gamma_{r3}(H))",
    "clique_number * restrained_domination_number": r"\omega(G) \cdot \gamma_r(H)",
    "clique_number + restrained_domination_number": r"(\omega(G) + \gamma_r(H))",
    "clique_number * outer_connected_domination_number": r"\omega(G) \cdot \tilde{\gamma}_c(H)",
    "clique_number + outer_connected_domination_number": r"(\omega(G) + \tilde{\gamma}_c(H))",


    "cartesian_product_chromatic_number": r"\chi(G \mathbin{\square} H)",
    "lexicographic_product_chromatic_number": r"\chi(G \circ H)",
    "tensor_product_chromatic_number": r"\chi(G \times H)",
    "strong_product_chromatic_number": r"\chi(G \boxtimes H)",

    "chromatic_number * domination_number": r"\chi(G) \cdot \gamma(H)",
    "chromatic_number + domination_number": r"(\chi(G) + \gamma(H))",
    "chromatic_number * total_domination_number": r"\chi(G) \cdot \gamma_t(H)",
    "chromatic_number + total_domination_number": r"(\chi(G) + \gamma_t(H))",
    "chromatic_number * independence_number": r"\chi(G) \cdot \alpha(H)",
    "chromatic_number + independence_number": r"(\chi(G) + \alpha(H))",
    "chromatic_number * matching_number": r"\chi(G) \cdot \mu(H)",
    "chromatic_number + matching_number": r"(\chi(G) + \mu(H))",
    "chromatic_number * edge_domination_number": r"\chi(G) \cdot \gamma_e(H)",
    "chromatic_number + edge_domination_number": r"(\chi(G) + \gamma_e(H))",
    "chromatic_number * vertex_cover_number": r"\chi(G) \cdot \beta(H)",
    "chromatic_number + vertex_cover_number": r"(\chi(G) + \beta(H))",
    "chromatic_number * clique_number": r"\chi(G) \cdot \omega(H)",
    "chromatic_number + clique_number": r"(\chi(G) + \omega(H))",
    "chromatic_number * chromatic_number": r"\chi(G) \cdot \chi(H)",
    "chromatic_number + chromatic_number": r"(\chi(G) + \chi(H))",
    "chromatic_number * residue": r"\chi(G) \cdot R(H)",
    "chromatic_number + residue": r"(\chi(G) + R(H))",
    "chromatic_number * annihilation_number": r"\chi(G) \cdot a(H)",
    "chromatic_number + annihilation_number": r"(\chi(G) + a(H))",
    "chromatic_number * slater": r"\chi(G) \cdot sl(H)",
    "chromatic_number + slater": r"(\chi(G) + sl(H))",
    "chromatic_number * roman_domination_number": r"\chi(G) \cdot \gamma_R(H)",
    "chromatic_number + roman_domination_number": r"(\chi(G) + \gamma_R(H))",
    "chromatic_number * double_roman_domination_number": r"\chi(G) \cdot \gamma_{dR}(H)",
    "chromatic_number + double_roman_domination_number": r"(\chi(G) + \gamma_{dR}(H))",
    "chromatic_number * two_rainbow_domination_number": r"\chi(G) \cdot \gamma_{r2}(H)",
    "chromatic_number + two_rainbow_domination_number": r"(\chi(G) + \gamma_{r2}(H))",
    "chromatic_number * three_rainbow_domination_number": r"\chi(G) \cdot \gamma_{r3}(H)",
    "chromatic_number + three_rainbow_domination_number": r"(\chi(G) + \gamma_{r3}(H))",
    "chromatic_number * restrained_domination_number": r"\chi(G) \cdot \gamma_r(H)",
    "chromatic_number + restrained_domination_number": r"(\chi(G) + \gamma_r(H))",
    "chromatic_number * outer_connected_domination_number": r"\chi(G) \cdot \tilde{\gamma}_c(H)",
    "chromatic_number + outer_connected_domination_number": r"(\chi(G) + \tilde{\gamma}_c(H))",


    "cartesian_product_residue": r"R(G \mathbin{\square} H)",
    "lexicographic_product_residue": r"R(G \circ H)",
    "tensor_product_residue": r"R(G \times H)",
    "strong_product_residue": r"R(G \boxtimes H)",

    "residue * domination_number": r"R(G) \cdot \gamma(H)",
    "residue + domination_number": r"(R(G) + \gamma(H))",
    "residue * total_domination_number": r"R(G) \cdot \gamma_t(H)",
    "residue + total_domination_number": r"(R(G) + \gamma_t(H))",
    "residue * independence_number": r"R(G) \cdot \alpha(H)",
    "residue + independence_number": r"(R(G) + \alpha(H))",
    "residue * matching_number": r"R(G) \cdot \mu(H)",
    "residue + matching_number": r"(R(G) + \mu(H))",
    "residue * edge_domination_number": r"R(G) \cdot \gamma_e(H)",
    "residue + edge_domination_number": r"(R(G) + \gamma_e(H))",
    "residue * vertex_cover_number": r"R(G) \cdot \beta(H)",
    "residue + vertex_cover_number": r"(R(G) + \beta(H))",
    "residue * clique_number": r"R(G) \cdot \omega(H)",
    "residue + clique_number": r"(R(G) + \omega(H))",
    "residue * chromatic_number": r"R(G) \cdot \chi(H)",
    "residue + chromatic_number": r"(R(G) + \chi(H))",
    "residue * residue": r"R(G) \cdot R(H)",
    "residue + residue": r"(R(G) + R(H))",
    "residue * annihilation_number": r"R(G) \cdot a(H)",
    "residue + annihilation_number": r"(R(G) + a(H))",
    "residue * slater": r"R(G) \cdot sl(H)",
    "residue + slater": r"(R(G) + sl(H))",
    "residue * roman_domination_number": r"R(G) \cdot \gamma_R(H)",
    "residue + roman_domination_number": r"(R(G) + \gamma_R(H))",
    "residue * double_roman_domination_number": r"R(G) \cdot \gamma_{dR}(H)",
    "residue + double_roman_domination_number": r"(R(G) + \gamma_{dR}(H))",
    "residue * two_rainbow_domination_number": r"R(G) \cdot \gamma_{r2}(H)",
    "residue + two_rainbow_domination_number": r"(R(G) + \gamma_{r2}(H))",
    "residue * three_rainbow_domination_number": r"R(G) \cdot \gamma_{r3}(H)",
    "residue + three_rainbow_domination_number": r"(R(G) + \gamma_{r3}(H))",
    "residue * restrained_domination_number": r"R(G) \cdot \gamma_r(H)",
    "residue + restrained_domination_number": r"(R(G) + \gamma_r(H))",
    "residue * outer_connected_domination_number": r"R(G) \cdot \tilde{\gamma}_c(H)",
    "residue + outer_connected_domination_number": r"(R(G) + \tilde{\gamma}_c(H))",


    "cartesian_product_annihilation_number": r"a(G \mathbin{\square} H)",
    "lexicographic_product_annihilation_number": r"a(G \circ H)",
    "tensor_product_annihilation_number": r"a(G \times H)",
    "strong_product_annihilation_number": r"a(G \boxtimes H)",

    "annihilation_number * domination_number": r"a(G) \cdot \gamma(H)",
    "annihilation_number + domination_number": r"(a(G) + \gamma(H))",
    "annihilation_number * total_domination_number": r"a(G) \cdot \gamma_t(H)",
    "annihilation_number + total_domination_number": r"(a(G) + \gamma_t(H))",
    "annihilation_number * independence_number": r"a(G) \cdot \alpha(H)",
    "annihilation_number + independence_number": r"(a(G) + \alpha(H))",
    "annihilation_number * matching_number": r"a(G) \cdot \mu(H)",
    "annihilation_number + matching_number": r"(a(G) + \mu(H))",
    "annihilation_number * edge_domination_number": r"a(G) \cdot \gamma_e(H)",
    "annihilation_number + edge_domination_number": r"(a(G) + \gamma_e(H))",
    "annihilation_number * vertex_cover_number": r"a(G) \cdot \beta(H)",
    "annihilation_number + vertex_cover_number": r"(a(G) + \beta(H))",
    "annihilation_number * clique_number": r"a(G) \cdot \omega(H)",
    "annihilation_number + clique_number": r"(a(G) + \omega(H))",
    "annihilation_number * chromatic_number": r"a(G) \cdot \chi(H)",
    "annihilation_number + chromatic_number": r"(a(G) + \chi(H))",
    "annihilation_number * residue": r"a(G) \cdot R(H)",
    "annihilation_number + residue": r"(a(G) + R(H))",
    "annihilation_number * annihilation_number": r"a(G) \cdot a(H)",
    "annihilation_number + annihilation_number": r"(a(G) + a(H))",
    "annihilation_number * slater": r"a(G) \cdot sl(H)",
    "annihilation_number + slater": r"(a(G) + sl(H))",
    "annihilation_number * roman_domination_number": r"a(G) \cdot \gamma_R(H)",
    "annihilation_number + roman_domination_number": r"(a(G) + \gamma_R(H))",
    "annihilation_number * double_roman_domination_number": r"a(G) \cdot \gamma_{dR}(H)",
    "annihilation_number + double_roman_domination_number": r"(a(G) + \gamma_{dR}(H))",
    "annihilation_number * two_rainbow_domination_number": r"a(G) \cdot \gamma_{r2}(H)",
    "annihilation_number + two_rainbow_domination_number": r"(a(G) + \gamma_{r2}(H))",
    "annihilation_number * three_rainbow_domination_number": r"a(G) \cdot \gamma_{r3}(H)",
    "annihilation_number + three_rainbow_domination_number": r"(a(G) + \gamma_{r3}(H))",
    "annihilation_number * restrained_domination_number": r"a(G) \cdot \gamma_r(H)",
    "annihilation_number + restrained_domination_number": r"(a(G) + \gamma_r(H))",
    "annihilation_number * outer_connected_domination_number": r"a(G) \cdot \tilde{\gamma}_c(H)",
    "annihilation_number + outer_connected_domination_number": r"(a(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_slater": r"sl(G \mathbin{\square} H)",
    "lexicographic_product_slater": r"sl(G \circ H)",
    "tensor_product_slater": r"sl(G \times H)",
    "strong_product_slater": r"sl(G \boxtimes H)",

    "slater * domination_number": r"sl(G) \cdot \gamma(H)",
    "slater + domination_number": r"(sl(G) + \gamma(H))",
    "slater * total_domination_number": r"sl(G) \cdot \gamma_t(H)",
    "slater + total_domination_number": r"(sl(G) + \gamma_t(H))",
    "slater * independence_number": r"sl(G) \cdot \alpha(H)",
    "slater + independence_number": r"(sl(G) + \alpha(H))",
    "slater * matching_number": r"sl(G) \cdot \mu(H)",
    "slater + matching_number": r"(sl(G) + \mu(H))",
    "slater * edge_domination_number": r"sl(G) \cdot \gamma_e(H)",
    "slater + edge_domination_number": r"(sl(G) + \gamma_e(H))",
    "slater * vertex_cover_number": r"sl(G) \cdot \beta(H)",
    "slater + vertex_cover_number": r"(sl(G) + \beta(H))",
    "slater * clique_number": r"sl(G) \cdot \omega(H)",
    "slater + clique_number": r"(sl(G) + \omega(H))",
    "slater * chromatic_number": r"sl(G) \cdot \chi(H)",
    "slater + chromatic_number": r"(sl(G) + \chi(H))",
    "slater * residue": r"sl(G) \cdot R(H)",
    "slater + residue": r"(sl(G) + R(H))",
    "slater * annihilation_number": r"sl(G) \cdot a(H)",
    "slater + annihilation_number": r"(sl(G) + a(H))",
    "slater * slater": r"sl(G) \cdot sl(H)",
    "slater + slater": r"(sl(G) + sl(H))",
    "slater * roman_domination_number": r"sl(G) \cdot \gamma_R(H)",
    "slater + roman_domination_number": r"(sl(G) + \gamma_R(H))",
    "slater * double_roman_domination_number": r"sl(G) \cdot \gamma_{dR}(H)",
    "slater + double_roman_domination_number": r"(sl(G) + \gamma_{dR}(H))",
    "slater * two_rainbow_domination_number": r"sl(G) \cdot \gamma_{r2}(H)",
    "slater + two_rainbow_domination_number": r"(sl(G) + \gamma_{r2}(H))",
    "slater * three_rainbow_domination_number": r"sl(G) \cdot \gamma_{r3}(H)",
    "slater + three_rainbow_domination_number": r"(sl(G) + \gamma_{r3}(H))",
    "slater * restrained_domination_number": r"sl(G) \cdot \gamma_r(H)",
    "slater + restrained_domination_number": r"(sl(G) + \gamma_r(H))",
    "slater * outer_connected_domination_number": r"sl(G) \cdot \tilde{\gamma}_c(H)",
    "slater + outer_connected_domination_number": r"(sl(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_roman_domination_number": r"\gamma_R(G \mathbin{\square} H)",
    "lexicographic_product_roman_domination_number": r"\gamma_R(G \circ H)",
    "tensor_product_roman_domination_number": r"\gamma_R(G \times H)",
    "strong_product_roman_domination_number": r"\gamma_R(G \boxtimes H)",

    "roman_domination_number * domination_number": r"\gamma_R(G) \cdot \gamma(H)",
    "roman_domination_number + domination_number": r"(\gamma_R(G) + \gamma(H))",
    "roman_domination_number * total_domination_number": r"\gamma_R(G) \cdot \gamma_t(H)",
    "roman_domination_number + total_domination_number": r"(\gamma_R(G) + \gamma_t(H))",
    "roman_domination_number * independence_number": r"\gamma_R(G) \cdot \alpha(H)",
    "roman_domination_number + independence_number": r"(\gamma_R(G) + \alpha(H))",
    "roman_domination_number * matching_number": r"\gamma_R(G) \cdot \mu(H)",
    "roman_domination_number + matching_number": r"(\gamma_R(G) + \mu(H))",
    "roman_domination_number * edge_domination_number": r"\gamma_R(G) \cdot \gamma_e(H)",
    "roman_domination_number + edge_domination_number": r"(\gamma_R(G) + \gamma_e(H))",
    "roman_domination_number * vertex_cover_number": r"\gamma_R(G) \cdot \beta(H)",
    "roman_domination_number + vertex_cover_number": r"(\gamma_R(G) + \beta(H))",
    "roman_domination_number * clique_number": r"\gamma_R(G) \cdot \omega(H)",
    "roman_domination_number + clique_number": r"(\gamma_R(G) + \omega(H))",
    "roman_domination_number * chromatic_number": r"\gamma_R(G) \cdot \chi(H)",
    "roman_domination_number + chromatic_number": r"(\gamma_R(G) + \chi(H))",
    "roman_domination_number * residue": r"\gamma_R(G) \cdot R(H)",
    "roman_domination_number + residue": r"(\gamma_R(G) + R(H))",
    "roman_domination_number * annihilation_number": r"\gamma_R(G) \cdot a(H)",
    "roman_domination_number + annihilation_number": r"(\gamma_R(G) + a(H))",
    "roman_domination_number * slater": r"\gamma_R(G) \cdot sl(H)",
    "roman_domination_number + slater": r"(\gamma_R(G) + sl(H))",
    "roman_domination_number * roman_domination_number": r"\gamma_R(G) \cdot \gamma_R(H)",
    "roman_domination_number + roman_domination_number": r"(\gamma_R(G) + \gamma_R(H))",
    "roman_domination_number * double_roman_domination_number": r"\gamma_R(G) \cdot \gamma_{dR}(H)",
    "roman_domination_number + double_roman_domination_number": r"(\gamma_R(G) + \gamma_{dR}(H))",
    "roman_domination_number * two_rainbow_domination_number": r"\gamma_R(G) \cdot \gamma_{r2}(H)",
    "roman_domination_number + two_rainbow_domination_number": r"(\gamma_R(G) + \gamma_{r2}(H))",
    "roman_domination_number * three_rainbow_domination_number": r"\gamma_R(G) \cdot \gamma_{r3}(H)",
    "roman_domination_number + three_rainbow_domination_number": r"(\gamma_R(G) + \gamma_{r3}(H))",
    "roman_domination_number * restrained_domination_number": r"\gamma_R(G) \cdot \gamma_r(H)",
    "roman_domination_number + restrained_domination_number": r"(\gamma_R(G) + \gamma_r(H))",
    "roman_domination_number * outer_connected_domination_number": r"\gamma_R(G) \cdot \tilde{\gamma}_c(H)",
    "roman_domination_number + outer_connected_domination_number": r"(\gamma_R(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_double_roman_domination_number": r"\gamma_{dR}(G \mathbin{\square} H)",
    "lexicographic_product_double_roman_domination_number": r"\gamma_{dR}(G \circ H)",
    "tensor_product_double_roman_domination_number": r"\gamma_{dR}(G \times H)",
    "strong_product_double_roman_domination_number": r"\gamma_{dR}(G \boxtimes H)",

    "double_roman_domination_number * domination_number": r"\gamma_{dR}(G) \cdot \gamma(H)",
    "double_roman_domination_number + domination_number": r"(\gamma_{dR}(G) + \gamma(H))",
    "double_roman_domination_number * total_domination_number": r"\gamma_{dR}(G) \cdot \gamma_t(H)",
    "double_roman_domination_number + total_domination_number": r"(\gamma_{dR}(G) + \gamma_t(H))",
    "double_roman_domination_number * independence_number": r"\gamma_{dR}(G) \cdot \alpha(H)",
    "double_roman_domination_number + independence_number": r"(\gamma_{dR}(G) + \alpha(H))",
    "double_roman_domination_number * matching_number": r"\gamma_{dR}(G) \cdot \mu(H)",
    "double_roman_domination_number + matching_number": r"(\gamma_{dR}(G) + \mu(H))",
    "double_roman_domination_number * edge_domination_number": r"\gamma_{dR}(G) \cdot \gamma_e(H)",
    "double_roman_domination_number + edge_domination_number": r"(\gamma_{dR}(G) + \gamma_e(H))",
    "double_roman_domination_number * vertex_cover_number": r"\gamma_{dR}(G) \cdot \beta(H)",
    "double_roman_domination_number + vertex_cover_number": r"(\gamma_{dR}(G) + \beta(H))",
    "double_roman_domination_number * clique_number": r"\gamma_{dR}(G) \cdot \omega(H)",
    "double_roman_domination_number + clique_number": r"(\gamma_{dR}(G) + \omega(H))",
    "double_roman_domination_number * chromatic_number": r"\gamma_{dR}(G) \cdot \chi(H)",
    "double_roman_domination_number + chromatic_number": r"(\gamma_{dR}(G) + \chi(H))",
    "double_roman_domination_number * residue": r"\gamma_{dR}(G) \cdot R(H)",
    "double_roman_domination_number + residue": r"(\gamma_{dR}(G) + R(H))",
    "double_roman_domination_number * annihilation_number": r"\gamma_{dR}(G) \cdot a(H)",
    "double_roman_domination_number + annihilation_number": r"(\gamma_{dR}(G) + a(H))",
    "double_roman_domination_number * slater": r"\gamma_{dR}(G) \cdot sl(H)",
    "double_roman_domination_number + slater": r"(\gamma_{dR}(G) + sl(H))",
    "double_roman_domination_number * roman_domination_number": r"\gamma_{dR}(G) \cdot \gamma_R(H)",
    "double_roman_domination_number + roman_domination_number": r"(\gamma_{dR}(G) + \gamma_R(H))",
    "double_roman_domination_number * double_roman_domination_number": r"\gamma_{dR}(G) \cdot \gamma_{dR}(H)",
    "double_roman_domination_number + double_roman_domination_number": r"(\gamma_{dR}(G) + \gamma_{dR}(H))",
    "double_roman_domination_number * two_rainbow_domination_number": r"\gamma_{dR}(G) \cdot \gamma_{r2}(H)",
    "double_roman_domination_number + two_rainbow_domination_number": r"(\gamma_{dR}(G) + \gamma_{r2}(H))",
    "double_roman_domination_number * three_rainbow_domination_number": r"\gamma_{dR}(G) \cdot \gamma_{r3}(H)",
    "double_roman_domination_number + three_rainbow_domination_number": r"(\gamma_{dR}(G) + \gamma_{r3}(H))",
    "double_roman_domination_number * restrained_domination_number": r"\gamma_{dR}(G) \cdot \gamma_r(H)",
    "double_roman_domination_number + restrained_domination_number": r"(\gamma_{dR}(G) + \gamma_r(H))",
    "double_roman_domination_number * outer_connected_domination_number": r"\gamma_{dR}(G) \cdot \tilde{\gamma}_c(H)",
    "double_roman_domination_number + outer_connected_domination_number": r"(\gamma_{dR}(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_two_rainbow_domination_number": r"\gamma_{r2}(G \mathbin{\square} H)",
    "lexicographic_product_two_rainbow_domination_number": r"\gamma_{r2}(G \circ H)",
    "tensor_product_two_rainbow_domination_number": r"\gamma_{r2}(G \times H)",
    "strong_product_two_rainbow_domination_number": r"\gamma_{r2}(G \boxtimes H)",

    "two_rainbow_domination_number * domination_number": r"\gamma_{r2}(G) \cdot \gamma(H)",
    "two_rainbow_domination_number + domination_number": r"(\gamma_{r2}(G) + \gamma(H))",
    "two_rainbow_domination_number * total_domination_number": r"\gamma_{r2}(G) \cdot \gamma_t(H)",
    "two_rainbow_domination_number + total_domination_number": r"(\gamma_{r2}(G) + \gamma_t(H))",
    "two_rainbow_domination_number * independence_number": r"\gamma_{r2}(G) \cdot \alpha(H)",
    "two_rainbow_domination_number + independence_number": r"(\gamma_{r2}(G) + \alpha(H))",
    "two_rainbow_domination_number * matching_number": r"\gamma_{r2}(G) \cdot \mu(H)",
    "two_rainbow_domination_number + matching_number": r"(\gamma_{r2}(G) + \mu(H))",
    "two_rainbow_domination_number * edge_domination_number": r"\gamma_{r2}(G) \cdot \gamma_e(H)",
    "two_rainbow_domination_number + edge_domination_number": r"(\gamma_{r2}(G) + \gamma_e(H))",
    "two_rainbow_domination_number * vertex_cover_number": r"\gamma_{r2}(G) \cdot \beta(H)",
    "two_rainbow_domination_number + vertex_cover_number": r"(\gamma_{r2}(G) + \beta(H))",
    "two_rainbow_domination_number * clique_number": r"\gamma_{r2}(G) \cdot \omega(H)",
    "two_rainbow_domination_number + clique_number": r"(\gamma_{r2}(G) + \omega(H))",
    "two_rainbow_domination_number * chromatic_number": r"\gamma_{r2}(G) \cdot \chi(H)",
    "two_rainbow_domination_number + chromatic_number": r"(\gamma_{r2}(G) + \chi(H))",
    "two_rainbow_domination_number * residue": r"\gamma_{r2}(G) \cdot R(H)",
    "two_rainbow_domination_number + residue": r"(\gamma_{r2}(G) + R(H))",
    "two_rainbow_domination_number * annihilation_number": r"\gamma_{r2}(G) \cdot a(H)",
    "two_rainbow_domination_number + annihilation_number": r"(\gamma_{r2}(G) + a(H))",
    "two_rainbow_domination_number * slater": r"\gamma_{r2}(G) \cdot sl(H)",
    "two_rainbow_domination_number + slater": r"(\gamma_{r2}(G) + sl(H))",
    "two_rainbow_domination_number * roman_domination_number": r"\gamma_{r2}(G) \cdot \gamma_R(H)",
    "two_rainbow_domination_number + roman_domination_number": r"(\gamma_{r2}(G) + \gamma_R(H))",
    "two_rainbow_domination_number * double_roman_domination_number": r"\gamma_{r2}(G) \cdot \gamma_{dR}(H)",
    "two_rainbow_domination_number + double_roman_domination_number": r"(\gamma_{r2}(G) + \gamma_{dR}(H))",
    "two_rainbow_domination_number * two_rainbow_domination_number": r"\gamma_{r2}(G) \cdot \gamma_{r2}(H)",
    "two_rainbow_domination_number + two_rainbow_domination_number": r"(\gamma_{r2}(G) + \gamma_{r2}(H))",
    "two_rainbow_domination_number * three_rainbow_domination_number": r"\gamma_{r2}(G) \cdot \gamma_{r3}(H)",
    "two_rainbow_domination_number + three_rainbow_domination_number": r"(\gamma_{r2}(G) + \gamma_{r3}(H))",
    "two_rainbow_domination_number * restrained_domination_number": r"\gamma_{r2}(G) \cdot \gamma_r(H)",
    "two_rainbow_domination_number + restrained_domination_number": r"(\gamma_{r2}(G) + \gamma_r(H))",
    "two_rainbow_domination_number * outer_connected_domination_number": r"\gamma_{r2}(G) \cdot \tilde{\gamma}_c(H)",
    "two_rainbow_domination_number + outer_connected_domination_number": r"(\gamma_{r2}(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_three_rainbow_domination_number": r"\gamma_{r3}(G \mathbin{\square} H)",
    "lexicographic_product_three_rainbow_domination_number": r"\gamma_{r3}(G \circ H)",
    "tensor_product_three_rainbow_domination_number": r"\gamma_{r3}(G \times H)",
    "strong_product_three_rainbow_domination_number": r"\gamma_{r3}(G \boxtimes H)",

    "three_rainbow_domination_number * domination_number": r"\gamma_{r3}(G) \cdot \gamma(H)",
    "three_rainbow_domination_number + domination_number": r"(\gamma_{r3}(G) + \gamma(H))",
    "three_rainbow_domination_number * total_domination_number": r"\gamma_{r3}(G) \cdot \gamma_t(H)",
    "three_rainbow_domination_number + total_domination_number": r"(\gamma_{r3}(G) + \gamma_t(H))",
    "three_rainbow_domination_number * independence_number": r"\gamma_{r3}(G) \cdot \alpha(H)",
    "three_rainbow_domination_number + independence_number": r"(\gamma_{r3}(G) + \alpha(H))",
    "three_rainbow_domination_number * matching_number": r"\gamma_{r3}(G) \cdot \mu(H)",
    "three_rainbow_domination_number + matching_number": r"(\gamma_{r3}(G) + \mu(H))",
    "three_rainbow_domination_number * edge_domination_number": r"\gamma_{r3}(G) \cdot \gamma_e(H)",
    "three_rainbow_domination_number + edge_domination_number": r"(\gamma_{r3}(G) + \gamma_e(H))",
    "three_rainbow_domination_number * vertex_cover_number": r"\gamma_{r3}(G) \cdot \beta(H)",
    "three_rainbow_domination_number + vertex_cover_number": r"(\gamma_{r3}(G) + \beta(H))",
    "three_rainbow_domination_number * clique_number": r"\gamma_{r3}(G) \cdot \omega(H)",
    "three_rainbow_domination_number + clique_number": r"(\gamma_{r3}(G) + \omega(H))",
    "three_rainbow_domination_number * chromatic_number": r"\gamma_{r3}(G) \cdot \chi(H)",
    "three_rainbow_domination_number + chromatic_number": r"(\gamma_{r3}(G) + \chi(H))",
    "three_rainbow_domination_number * residue": r"\gamma_{r3}(G) \cdot R(H)",
    "three_rainbow_domination_number + residue": r"(\gamma_{r3}(G) + R(H))",
    "three_rainbow_domination_number * annihilation_number": r"\gamma_{r3}(G) \cdot a(H)",
    "three_rainbow_domination_number + annihilation_number": r"(\gamma_{r3}(G) + a(H))",
    "three_rainbow_domination_number * slater": r"\gamma_{r3}(G) \cdot sl(H)",
    "three_rainbow_domination_number + slater": r"(\gamma_{r3}(G) + sl(H))",
    "three_rainbow_domination_number * roman_domination_number": r"\gamma_{r3}(G) \cdot \gamma_R(H)",
    "three_rainbow_domination_number + roman_domination_number": r"(\gamma_{r3}(G) + \gamma_R(H))",
    "three_rainbow_domination_number * double_roman_domination_number": r"\gamma_{r3}(G) \cdot \gamma_{dR}(H)",
    "three_rainbow_domination_number + double_roman_domination_number": r"(\gamma_{r3}(G) + \gamma_{dR}(H))",
    "three_rainbow_domination_number * two_rainbow_domination_number": r"\gamma_{r3}(G) \cdot \gamma_{r2}(H)",
    "three_rainbow_domination_number + two_rainbow_domination_number": r"(\gamma_{r3}(G) + \gamma_{r2}(H))",
    "three_rainbow_domination_number * three_rainbow_domination_number": r"\gamma_{r3}(G) \cdot \gamma_{r3}(H)",
    "three_rainbow_domination_number + three_rainbow_domination_number": r"(\gamma_{r3}(G) + \gamma_{r3}(H))",
    "three_rainbow_domination_number * restrained_domination_number": r"\gamma_{r3}(G) \cdot \gamma_r(H)",
    "three_rainbow_domination_number + restrained_domination_number": r"(\gamma_{r3}(G) + \gamma_r(H))",
    "three_rainbow_domination_number * outer_connected_domination_number": r"\gamma_{r3}(G) \cdot \tilde{\gamma}_c(H)",
    "three_rainbow_domination_number + outer_connected_domination_number": r"(\gamma_{r3}(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_restrained_domination_number": r"\gamma_r(G \mathbin{\square} H)",
    "lexicographic_product_restrained_domination_number": r"\gamma_r(G \circ H)",
    "tensor_product_restrained_domination_number": r"\gamma_r(G \times H)",
    "strong_product_restrained_domination_number": r"\gamma_r(G \boxtimes H)",

    "restrained_domination_number * domination_number": r"\gamma_r(G) \cdot \gamma(H)",
    "restrained_domination_number + domination_number": r"(\gamma_r(G) + \gamma(H))",
    "restrained_domination_number * total_domination_number": r"\gamma_r(G) \cdot \gamma_t(H)",
    "restrained_domination_number + total_domination_number": r"(\gamma_r(G) + \gamma_t(H))",
    "restrained_domination_number * independence_number": r"\gamma_r(G) \cdot \alpha(H)",
    "restrained_domination_number + independence_number": r"(\gamma_r(G) + \alpha(H))",
    "restrained_domination_number * matching_number": r"\gamma_r(G) \cdot \mu(H)",
    "restrained_domination_number + matching_number": r"(\gamma_r(G) + \mu(H))",
    "restrained_domination_number * edge_domination_number": r"\gamma_r(G) \cdot \gamma_e(H)",
    "restrained_domination_number + edge_domination_number": r"(\gamma_r(G) + \gamma_e(H))",
    "restrained_domination_number * vertex_cover_number": r"\gamma_r(G) \cdot \beta(H)",
    "restrained_domination_number + vertex_cover_number": r"(\gamma_r(G) + \beta(H))",
    "restrained_domination_number * clique_number": r"\gamma_r(G) \cdot \omega(H)",
    "restrained_domination_number + clique_number": r"(\gamma_r(G) + \omega(H))",
    "restrained_domination_number * chromatic_number": r"\gamma_r(G) \cdot \chi(H)",
    "restrained_domination_number + chromatic_number": r"(\gamma_r(G) + \chi(H))",
    "restrained_domination_number * residue": r"\gamma_r(G) \cdot R(H)",
    "restrained_domination_number + residue": r"(\gamma_r(G) + R(H))",
    "restrained_domination_number * annihilation_number": r"\gamma_r(G) \cdot a(H)",
    "restrained_domination_number + annihilation_number": r"(\gamma_r(G) + a(H))",
    "restrained_domination_number * slater": r"\gamma_r(G) \cdot sl(H)",
    "restrained_domination_number + slater": r"(\gamma_r(G) + sl(H))",
    "restrained_domination_number * roman_domination_number": r"\gamma_r(G) \cdot \gamma_R(H)",
    "restrained_domination_number + roman_domination_number": r"(\gamma_r(G) + \gamma_R(H))",
    "restrained_domination_number * double_roman_domination_number": r"\gamma_r(G) \cdot \gamma_{dR}(H)",
    "restrained_domination_number + double_roman_domination_number": r"(\gamma_r(G) + \gamma_{dR}(H))",
    "restrained_domination_number * two_rainbow_domination_number": r"\gamma_r(G) \cdot \gamma_{r2}(H)",
    "restrained_domination_number + two_rainbow_domination_number": r"(\gamma_r(G) + \gamma_{r2}(H))",
    "restrained_domination_number * three_rainbow_domination_number": r"\gamma_r(G) \cdot \gamma_{r3}(H)",
    "restrained_domination_number + three_rainbow_domination_number": r"(\gamma_r(G) + \gamma_{r3}(H))",
    "restrained_domination_number * restrained_domination_number": r"\gamma_r(G) \cdot \gamma_r(H)",
    "restrained_domination_number + restrained_domination_number": r"(\gamma_r(G) + \gamma_r(H))",
    "restrained_domination_number * outer_connected_domination_number": r"\gamma_r(G) \cdot \tilde{\gamma}_c(H)",
    "restrained_domination_number + outer_connected_domination_number": r"(\gamma_r(G) + \tilde{\gamma}_c(H))",

    "cartesian_product_outer_connected_domination_number": r"\tilde{\gamma}_c(G \mathbin{\square} H)",
    "lexicographic_product_outer_connected_domination_number": r"\tilde{\gamma}_c(G \circ H)",
    "tensor_product_outer_connected_domination_number": r"\tilde{\gamma}_c(G \times H)",
    "strong_product_outer_connected_domination_number": r"\tilde{\gamma}_c(G \boxtimes H)",

    "outer_connected_domination_number * domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma(H)",
    "outer_connected_domination_number + domination_number": r"(\tilde{\gamma}_c(G) + \gamma(H))",
    "outer_connected_domination_number * total_domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma_t(H)",
    "outer_connected_domination_number + total_domination_number": r"(\tilde{\gamma}_c(G) + \gamma_t(H))",
    "outer_connected_domination_number * independence_number": r"\tilde{\gamma}_c(G) \cdot \alpha(H)",
    "outer_connected_domination_number + independence_number": r"(\tilde{\gamma}_c(G) + \alpha(H))",
    "outer_connected_domination_number * matching_number": r"\tilde{\gamma}_c(G) \cdot \mu(H)",
    "outer_connected_domination_number + matching_number": r"(\tilde{\gamma}_c(G) + \mu(H))",
    "outer_connected_domination_number * edge_domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma_e(H)",
    "outer_connected_domination_number + edge_domination_number": r"(\tilde{\gamma}_c(G) + \gamma_e(H))",
    "outer_connected_domination_number * vertex_cover_number": r"\tilde{\gamma}_c(G) \cdot \beta(H)",
    "outer_connected_domination_number + vertex_cover_number": r"(\tilde{\gamma}_c(G) + \beta(H))",
    "outer_connected_domination_number * clique_number": r"\tilde{\gamma}_c(G) \cdot \omega(H)",
    "outer_connected_domination_number + clique_number": r"(\tilde{\gamma}_c(G) + \omega(H))",
    "outer_connected_domination_number * chromatic_number": r"\tilde{\gamma}_c(G) \cdot \chi(H)",
    "outer_connected_domination_number + chromatic_number": r"(\tilde{\gamma}_c(G) + \chi(H))",
    "outer_connected_domination_number * residue": r"\tilde{\gamma}_c(G) \cdot R(H)",
    "outer_connected_domination_number + residue": r"(\tilde{\gamma}_c(G) + R(H))",
    "outer_connected_domination_number * annihilation_number": r"\tilde{\gamma}_c(G) \cdot a(H)",
    "outer_connected_domination_number + annihilation_number": r"(\tilde{\gamma}_c(G) + a(H))",
    "outer_connected_domination_number * slater": r"\tilde{\gamma}_c(G) \cdot sl(H)",
    "outer_connected_domination_number + slater": r"(\tilde{\gamma}_c(G) + sl(H))",
    "outer_connected_domination_number * roman_domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma_R(H)",
    "outer_connected_domination_number + roman_domination_number": r"(\tilde{\gamma}_c(G) + \gamma_R(H))",
    "outer_connected_domination_number * double_roman_domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma_{dR}(H)",
    "outer_connected_domination_number + double_roman_domination_number": r"(\tilde{\gamma}_c(G) + \gamma_{dR}(H))",
    "outer_connected_domination_number * two_rainbow_domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma_{r2}(H)",
    "outer_connected_domination_number + two_rainbow_domination_number": r"(\tilde{\gamma}_c(G) + \gamma_{r2}(H))",
    "outer_connected_domination_number * three_rainbow_domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma_{r3}(H)",
    "outer_connected_domination_number + three_rainbow_domination_number": r"(\tilde{\gamma}_c(G) + \gamma_{r3}(H))",
    "outer_connected_domination_number * restrained_domination_number": r"\tilde{\gamma}_c(G) \cdot \gamma_r(H)",
    "outer_connected_domination_number + restrained_domination_number": r"(\tilde{\gamma}_c(G) + \gamma_r(H))",
    "outer_connected_domination_number * outer_connected_domination_number": r"\tilde{\gamma}_c(G) \cdot \tilde{\gamma}_c(H)",
    "outer_connected_domination_number + outer_connected_domination_number": r"(\tilde{\gamma}_c(G) + \tilde{\gamma}_c(H))",


}




DEF_MAP = {
    "order" : r"""The order of a graph $G$, denoted by $n(G)$, is the number of vertices in $G$.""",

    "size" : r"""The size of a graph $G$, denoted by $m(G)$, is the number of edges in $G$.""",

    "min_degree" : r"""The minimum degree of a graph $G$, denoted by $\delta(G)$, is the minimum
    degree of a vertex in $G$.""",

    "max_degree" : r"""The maximum degree of a graph $G$, denoted by $\Delta(G)$, is the maximum
    degree of a vertex in $G$.""",

    "diameter" : r"""The diameter of a graph $G$, denoted by $\text{diam}(G)$, is the maximum
    distance between any two vertices in $G$.""",

    "triameter": r""" The triameter of a graph $G$ is defined as
    $\text{tri}(G) = \max\{ d(u,v) + d(v,w) + d(u,w) \: : \: u,v,w \in V(G) \}$.""",

    "radius" : r"""The radius of a graph $G$, denoted by $\text{rad}(G)$, is the minimum distance
    between any two vertices in $G$.""",

    "domination_number": r"""A *dominating set* of $G$ is a set $D \subseteq V(G)$ of
    vertices such that every vertex in $G$ is either in $D$ or adjacent to a vertex in
    $D$. The *domination number* of a graph $G$, denoted by $\gamma(G)$, is the minimum
    cardinality of a dominating set of $G$.""",

    "total_domination_number" : r"""A *total dominating set* of $G$ is a set $D \subseteq V(G)$
    of vertices such that every vertex in $G$ is adjacent to a vertex in $D$. The *total domination number*
    of a graph $G$, denoted by $\gamma_t(G)$, is the minimum cardinality of a total dominating set of $G$. """,

    "connected_domination_number" : r"""A *connected dominating set* of $G$ is a dominating
    set $D \subseteq V(G)$ of vertices such that the subgraph induced by $D$ is connected.
    The *connected domination number* of a graph $G$, denoted by $\gamma_c(G)$, is the minimum
    cardinality of a connected dominating set of $G$.""",

    "independence_number": r"""An *independent set* of a graph $G$ is a set $I \subseteq V(G)$
    of vertices such that no two vertices in $I$ are adjacent. The *independence number* of $G$,
    denoted by $\alpha(G)$, is the maximum cardinality of an independent set of $G$.""",

    "chromatic_number": r"""The chromatic number of a graph $G$, denoted by $\chi(G)$, is
    the minimum number of colors needed to color the vertices of $G$ such that no two
    adjacent vertices have the same color.""",

    "square_chromatic_number": r"""The *square* of an undirected graph G is another graph denoted $G^2$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most 2. The chromatic number of the square of a graph $G$, denoted
    by $\chi(G^2)$, is the minimum number of colors needed to color the vertices of $G^2$ such
    that no two adjacent vertices have the same color.""",

    "clique_number": r"""A *clique* in $G$ is a set of vertices that induces a complete
    subgraph of $G$. The *clique number* of a graph $G$, denoted by $\omega(G)$, is the maximum cardinality of a clique in $G$.""",
    "vertex_cover_number": r"""A *vertex cover* of $G$ is a set $C \subseteq V(G)$ of vertices such that every edge in $G$ is incident to at least one
    vertex in $C$. The *vertex cover number* of a graph $G$, denoted by $\beta(G)$, is the minimum cardinality of a
    vertex cover of $G$.""",

    "zero_forcing_number": r"""Given a blue and white coloring of the vertices of $G$, the *zero forcing color change rule*
    allows any blue colored vertex with exactly one white colored neighbor to force its one white colored neighbor to become
    colored blue. A set $B \subseteq V(G)$ of initially blue colored vertices is called a *zero forcing set* of $G$ if by iteratively
    applying the zero forcing color change rule all of the vertices in $G$ become colored blue. The *zero forcing number* of
    a graph $G$, denoted by $Z(G)$, is the minimum cardinality of a zero forcing set of $G$.""",

    "total_zero_forcing_number": r"""Given a blue and white coloring of the vertices of $G$, the *zero forcing color change rule*
    allows any blue colored vertex with exactly one white colored neighbor to force its one white colored neighbor to become
    colored blue. A set $B \subseteq V(G)$ of initially blue colored vertices is called a *zero forcing set* of $G$ if by iteratively
    applying the zero forcing color change rule all of the vertices in $G$ become colored blue. The *total zero forcing number* of
    a graph $G$, denoted by $Z_t(G)$, is the minimum cardinality of a zero forcing set of $G$ which induces a isolate-free subgraph.""",

    "connected_zero_forcing_number": r"""Given a blue and white coloring of the vertices of $G$, the *zero forcing color change rule*
    allows any blue colored vertex with exactly one white colored neighbor to force its one white colored neighbor to become
    colored blue. A set $B \subseteq V(G)$ of initially blue colored vertices is called a *zero forcing set* of $G$ if by iteratively
    applying the zero forcing color change rule all of the vertices in $G$ become colored blue. The *connected zero forcing number* of
    a graph $G$, denoted by $Z_t(G)$, is the minimum cardinality of a zero forcing set of $G$ which induces a connected subgraph.""",

    "power_domination_number" : r"""Given a blue and white coloring of the vertices of $G$, the *zero forcing color change rule*
    allows any blue colored vertex with exactly one white colored neighbor to force its one white colored neighbor to become
    colored blue. A set $B \subseteq V(G)$ of initially blue colored vertices is called a *zero forcing set* of $G$ if by iteratively
    applying the zero forcing color change rule all of the vertices in $G$ become colored blue. A *power dominating set* of $G$ is
    a dominating set of a zero forcing set of $G$. The power domination number of a graph $G$, denoted by $\gamma_P(G)$, is the
    minimum cardinality of a power dominating set of $G$.""",

    "independent_domination_number" : r"""An independent dominating set of $G$ is a set $D \subseteq V(G)$ of vertices such that $D$
    is independent and every vertex in $G$ is adjacent to a vertex in $D$. The *independent domination number* of a graph $G$,
    denoted by $i(G)$, is the minimum cardinality of an independent dominating set of $G$. """,

    "matching_number" : r"""A *matching* in $G$ is a set of edges that do not share any common vertices. The *matching number*
    of a graph $G$, denoted by $\mu(G)$, is the maximum cardinality of a matching in $G$.""",

    "edge_domination_number": r"""An *edge dominating set* of $G$ is a set $D \subseteq E(G)$ of edges such that every
    edge in $G$ is incident to an edge in $D$. The *edge domination number* of a graph $G$, denoted by $\gamma_e(G)$,
    is the minimum cardinality of an edge dominating set of $G$.""",

    "annihilation_number" : r"""If $d_1, \dots, d_n$ is the degree sequence of a graph $G$ with $m$ edges, where $d_1 \leq \cdots \leq d_n$,
    then the *annihilation number* of $G$ is the largest integer $k$ such that $\sum_{i=1}^{k} d_i \leq m$, or equivalently, the largest integer
    $k$ such that $\sum_{i=1}^{k} d_i \leq \sum_{i=k+1}^{n} d_i$.""",

    "slater" : r"""The *Slater number* of a graph $G$, written $sl(G)$, is defined as the smallest integer $j$ such that
    $d_1 + \dots + d_j \geq n(G)$, where $d_1, \dots, d_n$ is the degree sequence of $G$ with $d_i \geq d_{i+1}$ for all $i \in [n(G) - 1]$.""",

    "sub_total_domination_number" : r"""The sub-total domination number of a graph $G$ is denoted by $\text{sub}_t(G)$,
    smallest integer $j$ such that
    $j + (d_1 + \dots + d_j) \geq n(G)$, where $d_1, \dots, d_n$ is the degree sequence of $G$ with $d_i \geq d_{i+1}$ for all $i \in [n(G) - 1]$.""",

    "wiener_index" : r"""The *Wiener index* of a graph $G$, denoted by $W(G)$, is the sum of the shortest path distances between all pairs of vertices in $G$.
    Formally, if $d(u, v)$ denotes the distance between vertices $u$ and $v$ in $G$, then $W(G) = \sum_{\{u,v\} \subseteq V(G)} d(u, v).$""",


    "two_rainbow_domination_number": r"""Given a graph $G$ and a set of $[k] = \{1, \dots, k\}$ colors, assume that we assign
    an arbitrary subset of these colors to each vertex of $G$. If we require that each vertex to which an empty set is assigned
    has in its (open) neighborhood all $k$ colors, then this assignment is called a *$k$-rainbow dominating function* of the graph $G$.
    The corresponding invariant $\gamma_{rk}(G)$, which is the minimum sum of numbers of assigned colors over all vertices of $G$,
    is called the *$k$-rainbow domination number* of G. The *2-rainbow domination number* of $G$, denoted
    $\gamma_{r2}(G)$, is the minimum cardinality of a 2-rainbow dominating set of $G$.""",

    "roman_domination_number": r"""A *Roman dominating function* of $G$ is a function $f \: : \: V(G) \rightarrow \{0, 1, 2\}$
    such that every vertex $v$ for which $f(v) = 0$ has a neighbor $u$ with $f(u) = 2$. The weight of a Roman dominating function
    $f$ is $w(f) = \sum_{v \in V(G)} f(v)$. The *Roman domination number* of a graph $G$, denoted by $\gamma_R(G)$, is the minimum
    weight of all possible Roman dominating functions in $G$.""",

    "three_rainbow_domination_number": r"""Given a graph $G$ and a set of $[k] = \{1, \dots, k\}$ colors, assume that we assign
    an arbitrary subset of these colors to each vertex of $G$. If we require that each vertex to which an empty set is assigned
    has in its (open) neighborhood all $k$ colors, then this assignment is called a *$k$-rainbow dominating function* of the graph $G$.
    The corresponding invariant $\gamma_{rk}(G)$, which is the minimum sum of numbers of assigned colors over all vertices of $G$,
    is called the *$k$-rainbow domination number* of G. The *3-rainbow domination number* of $G$, denoted
    $\gamma_{r3}(G)$, is the minimum cardinality of a 3-rainbow dominating set in $G$.""",

    "double_roman_domination_number": r"""A function $f \: : \: V(G) \rightarrow \{0, 1, 2, 3\}$ is a double Roman dominating
    function on a graph $G$ if the following conditions are met. Let $V_i$ denote the set of vertices assigned $i$ by function
    $f$. (i) If $f(v) = 0 $, then vertex $v$ must have at least two neighbors in $V_2$ or one neighbor in $V_3$. (ii) If $f(v) = 1$,
    then vertex $v$ must have at least one neighbor in $V_2 \cup V_3$. The *double Roman domination number* of $G$, written
    $\gamma_{dR}(G)$, equals the minimum weight of a double Roman dominating function on $G$. """,

    "restrained_domination_number": r"""A *restrained dominating set* of $G$, is a dominating set $D \subseteq V(G)$,
    such that each vertex in $V(G) \setminus D$ is adjacent to another vertex in $V(G) \setminus D$.
    The *restrained domination number* of a graph $G$, denoted by $\gamma_{r}(G)$, is the minimum
    cardinality of a restrained dominating set of $G$.""",

    "residue_residue_power_sum": r"""The *$k$-power* of an undirected graph $G$ is another graph denoted $G^k$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most $k$. The *residue* of a graph is the number of zero obtained after termination
    of the Havel-Hakimi process on $G$, and is denoted $R(G)$.""",

    "power_max_degree_residue_sum": r"""The *$k$-power* of an undirected graph $G$ is another graph denoted $G^k$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most $k$. The *maximum degree* of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.
    The *residue* of a graph is the number of zero obtained after termination of the Havel-Hakimi process on $G$, and is denoted $R(G)$.""",

    "power_max_degree_annihilation_sum": r"""The *$k$-power* of an undirected graph $G$ is another graph denoted $G^k$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most $k$. The *maximum degree* of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.
    If $d_1, \dots, d_n$ is the degree sequence of a graph $G$ with $m$ edges, where $d_1 \leq \cdots \leq d_n$,
    then the *annihilation number* of $G$ is the largest integer $k$ such that $\sum_{i=1}^{k} d_i \leq m$, or equivalently, the largest integer
    $k$ such that $\sum_{i=1}^{k} d_i \leq \sum_{i=k+1}^{n} d_i$.""",

    "power_min_degree_residue_sum": r"""The sum of the residues of $G$ through $G^{\delta}$.""",

    "power_min_degree_annihilation_sum": r"""The *$k$-power* of an undirected graph $G$ is another graph denoted $G^k$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most $k$. The *minimum degree* of a graph $G$, denoted by $\delta(G)$, is the maximum degree of a vertex in $G$.
    If $d_1, \dots, d_n$ is the degree sequence of a graph $G$ with $m$ edges, where $d_1 \leq \cdots \leq d_n$,
    then the *annihilation number* of $G$ is the largest integer $k$ such that $\sum_{i=1}^{k} d_i \leq m$, or equivalently, the largest integer
    $k$ such that $\sum_{i=1}^{k} d_i \leq \sum_{i=k+1}^{n} d_i$.""",

    "cubed_chromatic_number": r"""The *cube* of an undirected graph G is another graph denoted $G^3$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most 3. The chromatic number of the cube of a graph $G$, denoted by
    $\chi(G^3)$, is the minimum number of colors needed to color the vertices of the cube $G^3$ of $G$ such
    that no two adjacent vertices have the same color.""",

    "cube_residue": r"""The *cube* of an undirected graph G is another graph denoted $G^3$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most 3. The residue of a graph $G$, denoted by $R(G)$, is the
    number of zeros at the termination of the Havel-Hakimi proccess on the degree sequence of $G$.""",

    "cube_annihilation": r"""The *cube* of an undirected graph G is another graph denoted $G^3$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most 3. If $d_1, \dots, d_n$ is the degree sequence of a graph $G$ with $m$ edges, where $d_1 \leq \cdots \leq d_n$,
    then the *annihilation number* of $G$ is the largest integer $k$ such that $\sum_{i=1}^{k} d_i \leq m$, or equivalently, the largest integer
    $k$ such that $\sum_{i=1}^{k} d_i \leq \sum_{i=k+1}^{n} d_i$.""",

    "square_clique_number": r""" The square of an undirected graph G is another graph denoted $G^2$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most 2. The clique number of the square of a graph $G$, denoted by $\omega(G^2)$,
    is the maximum cardinality of a clique in $G^2$.""",

    "LG_residue": r"""The residue of a graph $G$, denoted by $R(G)$, is the number of zeros at the
    termination of the Havel-Hakimi proccess on the degree sequence of $G$. The line graph of a graph $G$,
    denoted by $L(G)$, is the graph whose vertices correspond to the edges of $G$ and two vertices in $L(G)$
    are adjacent if their corresponding edges in $G$ share a common vertex.""",

    "LG_annihilation": r"""The annihilation number of a graph $G$, denoted by $a(G)$, is a degree sequence invariant introduced by R. Pepper. The line graph of a graph $G$, denoted by $L(G)$, is the graph whose vertices correspond to the edges of $G$ and two vertices in $L(G)$ are adjacent if their corresponding edges in $G$ share a common vertex.""",
    "semitotal_domination_number": r"""A *semitotal dominating set* is a set of vertices $S \subseteq V(G)$ in $G$ such that $S$ is a dominating set and each vertex of $S$ within distance 2 to another vertex in $S$.
    The *semitotal domination number* of $G$ is the minimum cardinality of a semitotal dominating set of $G$, and is denoted by $\gamma_{2t}(G)$.""",

    "power_2_residue_sum": r"""The square of an undirected graph G is another graph denoted $G^2$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most 2. The sum of the residue of a graph $G$ and the residue of the square of $G$.""",
    "power_3_residue_sum": r"""The sum of the residue of a graph $G$, the residue of the square of $G$, and the residue of the cube of $G$.""",
    "sum_connectivity_index": r"""The sum connectivity index of a graph $G$ is a degree sequence graph invariant denoted by $\text{sum}_c(G)$.""",
    "power_2_annihilation_sum": r"""The square of an undirected graph G is another graph denoted $G^2$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$
    is at most 2. The sum of the annihilation number of a graph $G$ and the annihilation number of the square of $G$.""",
    "power_3_annihilation_sum": r"""The sum of the annihilation number of a graph $G$, the annihilation number of the square of $G$, and the annihilation number of the cube of $G$.""",
    "min_edge_cover": r"""A *minimum edge cover* of $G$ is a set $E \subseteq E(G)$ of edges such that every vertex in $G$ is incident to an edge in $E$. The *minimum edge cover number* of a graph $G$, denoted by $\beta'(G)$, is the minimum cardinality of a minimum edge cover of $G$.""",
    "randic_index": r"""The Randić index of a graph $G$ is a degree sequence graph invariant denoted by $\text{randic}(G)$.""",

    "square_positive_energy" : r"""The square positive energy of a graph $G$, denoted $[s^{+}(G)]$, is
    the sum of the squares of the eigenvalues of the adjacency matrix of $G$ *rounded to the nearest integer*.""",

    "square_negative_energy" : r"""The square negative energy of a graph $G$, denoted $[s^{-}(G)]$, is the
    sum of the squares of the negative eigenvalues of the adjacency matrix of $G$ *rounded to the nearest integer*.""",

    "graph_energy" : r"""The energy of a graph $G$, denoted $[\mathcal{E}(G)]$, is the sum of the absolute values
    of the eigenvalues of the adjacency matrix of $G$ *rounded to the nearest integer*.""",

    "second_largest_eigenvalue" : r"""The second largest eigenvalue of a graph $G$, denoted by $[\lambda_2(G)]$, is
    the second largest eigenvalue of the adjacency matrix of $G$ *rounded to the nearest integer*.""",

    "min_maximal_matching_number" : r"""The minimum maximal matching number of a graph $G$, denoted by $i(L(G))$,
    is the minimum cardinality of a maximal matching in $G$; equivalently, the independent domination number of
    the line graph L(G).""",

    "positive_semidefinite_zero_forcing_number" : r"""In a graph $G$ where some vertices $S$ are colored blue and
    the remaining vertices are colored white, the positive semidefinite color change rule is: If $W_1, \dots, W_k$
    are the sets of vertices of the $k$ components of $G - S$ (note that it is possible that $k=1$), $w \in W_i$,
    $u \in S$, and $w$ is the only white neighbor of $u$ in the subgraph of $G$ induced by $W_i \cup S$, then change
    the color of $w$ to blue; in this case, we say $u$ forces $w$ and write $u \to w$. Given an initial set $B$ of
    blue vertices, the derived set of $B$ is the set of blue vertices that results from applying the positive
    semidefinite color change rule until no more changes are possible. A positive semidefinite zero forcing set
    is an initial set $B$ of vertices such that the derived set of $B$ is all the vertices of $G$. The
    *positive semidefinite zero forcing number* of a graph $G$, denoted $Z_+(G)$, is the minimum of $|B|$ over all
    positive semidefinite zero forcing sets $B \subseteq V(G)$.""",

    "residue" : r"""The residue of a graph $G$, denoted by $R(G)$, is the number of zeros at the termination of the Havel-Hakimi proccess on
    the degree sequence of $G$.""",

    "harmonic_index" : r"""The *harmonic index* of a graph $G$, denoted by $H(G)$, is a degree-based graph invariant defined as the sum of the reciprocals of the degrees of the adjacent vertices. Formally, if $d(u)$ and $d(v)$ are the degrees of adjacent vertices $u$ and $v$, then
    $$H(G) = \sum_{\{u,v\} \in E(G)} \frac{2}{d(u) + d(v)}.$$""",


    "square_zero_forcing_number" : r"""Given a blue and white coloring of the vertices of $G$, the *zero forcing color change rule*
    allows any blue colored vertex with exactly one white colored neighbor to force its one white colored neighbor to become
    colored blue. A set $B \subseteq V(G)$ of initially blue colored vertices is called a *zero forcing set* of $G$ if by iteratively
    applying the zero forcing color change rule all of the vertices in $G$ become colored blue. The *zero forcing number* of
    a graph $G$, denoted by $Z(G)$, is the minimum cardinality of a zero forcing set of $G$. The square of an undirected graph G is another graph denoted $G^2$
    that has the same set of vertices, but in which two vertices are adjacent when their distance in $G$ is at most 2.""",

    "LG_graph_energy" : r"""The energy of the line graph of a graph $G$, denoted by $[\mathcal{E}(L(G))]$, is
    the sum of the absolute values of the eigenvalues of the adjacency matrix of $L(G)$ *rounded to the nearest integer*.""",

    "LG_slater" : r"""The Slater number of the line graph of a graph $G$ is a degree sequence graph invariant denoted by $sl(L(G))$.""",

    "k_slater_index": r"""The $k$-Slater index of a graph $G$, denoted by $sl(G, k)$, is the smallest integer $k \geq 1$ so that $\text{sub}_k(G) \geq \gamma(G)$,
     where $\text{sub}_k(G)$ is the sub-$k$-domination number of $G$.""",

    "k_residual_index" : r"""The $k$-residual index of a graph $G$, denoted by $R(G, k)$, is the smallest integer $k \geq 1$ so that $\text{R}_k(G) \geq \alpha(G)$,
    where $\text{R}_k(G)$ is the $k$-residue of $G$ and $\alpha(G)$ is the independence number of $G$.""",

    "(order - domination_number)":  r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The domination number of a graph $G$, denoted by $\gamma(G)$, is the minimum cardinality of a dominating set of $G$.
    A dominating set of $G$ is a set $D \subseteq V(G)$ of vertices such that every vertex in $G$ is either in $D$ or adjacent to a vertex in $D$.""",
    "(order - total_domination_number)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The total domination number of a graph $G$,
    denoted by $\gamma_t(G)$, is the minimum cardinality of a total dominating set of $G$. A total dominating set of $G$ is a set $D \subseteq V(G)$ of vertices such that every vertex in $G$ is adjacent to a vertex in $D$.""",
    "(order - connected_domination_number)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The connected domination number of a graph $G$,
    denoted by $\gamma_c(G)$, is the minimum cardinality of a connected dominating set of $G$. A connected dominating set of $G$ is a dominating set $D \subseteq V(G)$ of vertices such that the subgraph induced by $D$ is connected.""",
    "(order - power_domination_number)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The power domination number of a graph $G$, denoted by $\gamma_P(G)$,
    is the minimum cardinality of a power dominating set of $G$.""",
    "(order - zero_forcing_number)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The zero forcing number of a graph $G$, denoted by $Z(G)$, is the minimum size of a zero forcing set of $G$.
    A zero forcing set of $G$ is a set $S \subseteq V(G)$ of vertices such that if the vertices in $S$ are initially colored blue and all other vertices are initially colored white, then the coloring process will eventually turn all vertices blue.""",
    "(order - diameter)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The diameter of a graph $G$, denoted by $\text{diam}(G)$, is the maximum distance between any two vertices in $G$.""",
    "(order - radius)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The radius of a graph $G$, denoted by $\text{rad}(G)$, is the minimum distance between any two vertices in $G$.""",
    "(order - triameter)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The triameter of a graph $G$is denoted by $\text{tri}(G)$.""",
    "(order - independent_domination_number)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The independent domination number of a graph $G$, denoted by $i(G)$, is the minimum cardinality of an independent dominating set of $G$.
    An independent dominating set of $G$ is a set $D \subseteq V(G)$ of vertices such that $D$ is independent and every vertex in $G$ is adjacent to a vertex in $D$.""",
    "(order - chromatic_number)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The chromatic number of a graph $G$, denoted by $\chi(G)$, is the minimum number of colors needed to color the
    vertices of $G$ such that no two adjacent vertices have the same color.""",
    "square_residue": r"""The square residue of a graph $G$, denoted by $R(G^2)$, is the number of zeros at the termination of the Havel-Hakimi proccess on the degree sequence of $G^2$.""",
    "square_annihilation": r"""The square annihilation number of a graph $G$, denoted by $a(G^2)$, is a degree sequence invariant introduced by R. Pepper.""",
    "(order - matching_number)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The matching number of a graph $G$, denoted by $\mu(G)$, is the maximum cardinality of a matching in $G$.
    A matching in $G$ is a set of edges that do not share any common vertices.""",
    "(order - min_maximal_matching_number)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The minimum maximal matching number of a graph $G$, denoted by $\gamma_e(G)$, is the minimum cardinality of a maximal matching in $G$.""",
    "(order - min_degree)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The minimum degree of a graph $G$, denoted by $\delta(G)$, is the minimum degree of a vertex in $G$.""",
    "(order - max_degree)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The maximum degree of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.""",
    "(order - clique_number)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The clique number of a graph $G$, denoted by $\omega(G)$, is the maximum cardinality of a clique in $G$.
    A clique in $G$ is a set of vertices that induces a complete subgraph of $G$.""",
    "(order - residue)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The residue of a graph $G$, denoted by $R(G)$, is the number of zeros at the termination of the Havel-Hakimi proccess on the degree sequence of $G$.""",
    "(order - annihilation_number)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The annihilation number of a graph $G$, denoted by $a(G)$, is a degree sequence invariant introduced by R. Pepper.""",
    "(order - sub_total_domination_number)" : r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The sub-total domination number of a graph $G$ is denoted by $\text{sub}_t(G)$.""",
    "(order - slater)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The Slater number of a graph $G$ is a degree sequence graph invariant denoted by $sl(G)$.""",
    "(order - k_slater_index)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The $k$-Slater index of a graph $G$, denoted by $sl(G, k)$, is the smallest integer $k \geq 1$ so that $\text{sub}_k(G) \geq \gamma(G)$.""",
    "(order - k_residual_index)": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The $k$-residual index of a graph $G$, denoted by $R(G, k)$, is the smallest integer $k \geq 1$ so that $\text{R}_k(G) \geq \alpha(G)$.""",
    "[(annihilation_number + residue)/ max_degree]": r"""The annihilation number of a graph $G$, denoted by $a(G)$, is a degree sequence invariant introduced by R. Pepper. The residue of a graph $G$, denoted by $R(G)$, is the number of zeros at the termination of the Havel-Hakimi proccess on the degree sequence of $G$. The maximum degree of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.""",
    "[order/ max_degree]": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The maximum degree of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.""",
    "[order/ (max_degree + 1)]": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The maximum degree of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.""",
    "[order/ (max_degree - 1)]": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The maximum degree of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.""",
    "[order/ (max_degree + 2)]": r"""The order of $G$, denoted $n(G)$, is the number of vertices in $G$. The maximum degree of a graph $G$, denoted by $\Delta(G)$, is the maximum degree of a vertex in $G$.""",
    "(residue + annihilation_number)": r"""The residue of a graph $G$, denoted by $R(G)$, is the number of zeros at the termination of the Havel-Hakimi proccess on the degree sequence of $G$. The annihilation number of a graph $G$, denoted by $a(G)$, is a degree sequence invariant introduced by R. Pepper.""",
    "a connected graph": r"""A connected graph is a graph in which there is a path between every pair of vertices.""",
    "a connected and bipartite graph": r"""A connected and bipartite graph is a graph in which the vertices can be partitioned into two sets such that
    no two vertices in the same set are adjacent.""",
    "an eulerian graph": r"""An *Eulerian trail* (or *Eulerian path*) is a trail in a finite graph that visits every edge exactly once (allowing for revisiting vertices).
    Similarly, an *Eulerian circuit* or *Eulerian cycle* is an Eulerian trail that starts and ends on the same vertex. The term *Eulerian graph* has two common meanings in graph theory.
    One meaning is a graph with an Eulerian circuit, and the other is a graph with every vertex of even degree. These definitions coincide for connected graphs.""",
    "a connected and planar graph": r"""A *planar graph* is a graph that can be embedded in the plane without any edges crossing.""",
    "a connected and regular graph": r"""A connected and $r$-regular graph with $r>0$ is a non-trivial graph in which every vertex has degree $r$.""",
    "a connected and cubic graph": r"""A connected and cubic graph is a graph in which every vertex has degree 3.""",
    "a connected graph which is not K_n": r"""A connected graph which is not $K_n$ is a graph with $n$ vertices that is not a complete graph.""",
    "a connected and triangle-free graph": r"""A connected and triangle-free graph is a graph in which no three vertices form a triangle.""",
    "a connected and at-free graph": r"""A connected graph is a graph in which there is a path between every pair of vertices. Three vertices of a graph form an *asteroidal triple*
    if every two of them are connected by a path avoiding the neighborhood of the third. A graph is *at-free* if it does not contain any asteroidal triple.""",
    "a tree graph": r"""A tree is a connected graph with no cycles.""",
    "outer_connected_domination_number": r"""The outer connected domination number of a graph $G$, denoted by $\tilde{\gamma}_{c}(G)$, is the minimum cardinality of a dominating set $S$ of $G$ such that $V(G)\setminus S$ induces a connected subgraph.""",
    "a connected and chordal graph": r"""A *chordal graph* is one in which all cycles of four or more vertices have a *chord*, which is an edge that is not part of the cycle but connects two
    vertices of the cycle. Equivalently, every induced cycle in the graph should have exactly three vertices.""",
    "a connected and claw-free graph": r"""A *claw* is another name for the complete bipartite graph $K_{1,3}$ (that is, a star graph comprising three edges, three leaves, and a central vertex).
    A *claw-free graph* is a graph in which no induced subgraph is a claw; i.e., any subset of four vertices has other than only three edges connecting them in this pattern. Equivalently,
    a claw-free graph is a graph in which the neighborhood of any vertex is the complement of a triangle-free graph.""",
    "a connected graph with maximum degree at most 3": r"""A connected graph with maximum degree at most 3 is a graph in which every vertex has degree at most 3.""",
    "a connected graph which is not K_n and has maximum degree at most 3": r"""A connected graph with maximum degree at most 3 is a graph in which every vertex has degree at most 3.""",
    "a connected, claw-free, and cubic graph": r"""A *cubic graph* is a graph where every vertex has degree 3. A *claw* is another name for the complete bipartite graph $K_{1,3}$ (that is, a star graph comprising three edges, three leaves, and a central vertex).
    A *claw-free graph* is a graph in which no induced subgraph is a claw; i.e., any subset of four vertices has other than only three edges connecting them in this pattern. Equivalently,
    a claw-free graph is a graph in which the neighborhood of any vertex is the complement of a triangle-free graph.""",
    "a connected, planar, and cubic graph": r"""A connected, planar, and cubic graph is a graph in which every vertex has degree 3 and can be embedded in the plane without any edges crossing.""",
    "a connected and cubic graph which is not K_4": r"""A connected and cubic graph which is not $K_4$ is a graph in which every vertex has degree 3 and is not a complete graph on 4 vertices.""",
    "a connected and well-covered graph": r"""A connected and well-covered graph is a graph in which every maximal independent set is also maximum.""",
    "a connected graph with diameter at most 3": r"""A connected graph with diameter at most 3 is a graph in which the maximum distance between any two vertices is at most 3.""",
    "a connected and planar graph with diameter at most 3": r"""A connected and planar graph with diameter at most 3 is a graph that can be embedded in the plane without any edges crossing and the maximum distance between any two vertices is at most 3.""",
    "a connected graph with a total domination number equal to the domination number": r"""A connected graph with a total domination number equal to the domination number is a graph in which the minimum cardinality of a total dominating set is equal to the minimum cardinality of a dominating set.""",
    "a connected_graph with min_degree at least 2": r"""A connected graph with minimum degree at least 2 is a graph in which every vertex has degree at least 2.""",
    "a connected_graph with min_degree at least 3": r"""A connected graph with minimum degree at least 3 is a graph in which every vertex has degree at least 3.""",
    "a connected_graph with min_degree at least 4": r"""A connected graph with minimum degree at least 4 is a graph in which every vertex has degree at least 4.""",
    "a connected and bipartite graph with min_degree at least 2": r"""A connected and bipartite graph with minimum degree at least 2 is a graph in which the vertices can be partitioned into two sets such that no two vertices in the same set are adjacent and every vertex has degree at least 2.""",
    "a connected and bipartite graph with min_degree at least 3": r"""A connected and bipartite graph with minimum degree at least 3 is a graph in which the vertices can be partitioned into two sets such that no two vertices in the same set are adjacent and every vertex has degree at least 3.""",
    "a connected and bipartite graph with min_degree at least 4": r"""A connected and bipartite graph with minimum degree at least 4 is a graph in which the vertices can be partitioned into two sets such that no two vertices in the same set are adjacent and every vertex has degree at least 4.""",
    "a connected and planar graph with min_degree at least 2": r"""A connected and planar graph with minimum degree at least 2 is a graph that can be embedded in the plane without any edges crossing and every vertex has degree at least 2.""",
    "a connected and planar graph with min_degree at least 3": r"""A connected and planar graph with minimum degree at least 3 is a graph that can be embedded in the plane without any edges crossing and every vertex has degree at least 3.""",
    "a connected and planar graph with min_degree at least 4": r"""A connected and planar graph with minimum degree at least 4 is a graph that can be embedded in the plane without any edges crossing and every vertex has degree at least 4.""",
    "a connected graph which is not K_n with min_degree at least 2": r"""A connected graph which is not $K_n$ with minimum degree at least 2 is a graph with $n$ vertices that is not a complete graph and every vertex has degree at least 2.""",
    "a connected graph which is not K_n with min_degree at least 3": r"""A connected graph which is not $K_n$ with minimum degree at least 3 is a graph with $n$ vertices that is not a complete graph and every vertex has degree at least 3.""",
    "a connected graph which is not K_n with min_degree at least 4": r"""A connected graph which is not $K_n$ with minimum degree at least 4 is a graph with $n$ vertices that is not a complete graph and every vertex has degree at least 4.""",
    "a connected and triangle-free graph with min_degree at least 2": r"""A connected and triangle-free graph with minimum degree at least 2 is a graph in which no three vertices form a triangle and every vertex has degree at least 2.""",
    "a connected and triangle-free graph with min_degree at least 3": r"""A connected and triangle-free graph with minimum degree at least 3 is a graph in which no three vertices form a triangle and every vertex has degree at least 3.""",
    "a connected and triangle-free graph with min_degree at least 4": r"""A connected and triangle-free graph with minimum degree at least 4 is a graph in which no three vertices form a triangle and every vertex has degree at least 4.""",
    "a connected and at-free graph with min_degree at least 2": r"""A connected and at-free graph with minimum degree at least 2 is a graph in which every vertex has degree at least 2 and does not contain any asteroidal triple.""",
    "a connected and at-free graph with min_degree at least 3": r"""A connected and at-free graph with minimum degree at least 3 is a graph in which every vertex has degree at least 3 and does not contain any asteroidal triple.""",
    "a connected and at-free graph with min_degree at least 4": r"""A connected and at-free graph with minimum degree at least 4 is a graph in which every vertex has degree at least 4 and does not contain any asteroidal triple.""",
    "a connected and chordal graph with min_degree at least 2": r"""A connected and chordal graph with minimum degree at least 2 is a graph in which every vertex has degree at least 2 and every induced cycle has exactly three vertices.""",
    "a connected and chordal graph with min_degree at least 3": r"""A connected and chordal graph with minimum degree at least 3 is a graph in which every vertex has degree at least 3 and every induced cycle has exactly three vertices.""",
    "a connected and chordal graph with min_degree at least 4": r"""A connected and chordal graph with minimum degree at least 4 is a graph in which every vertex has degree at least 4 and every induced cycle has exactly three vertices.""",
    "a connected and claw-free graph with min_degree at least 2": r"""A connected and claw-free graph with minimum degree at least 2 is a graph in which every vertex has degree at least 2 and does not contain any claw.""",
    "a connected and claw-free graph with min_degree at least 3": r"""A connected and claw-free graph with minimum degree at least 3 is a graph in which every vertex has degree at least 3 and does not contain any claw.""",
    "a connected and claw-free graph with min_degree at least 4": r"""A connected and claw-free graph with minimum degree at least 4 is a graph in which every vertex has degree at least 4 and does not contain any claw.""",
    "a connected graph with min_degree at least 2 and maximum degree at most 3": r"""A connected graph with minimum degree at least 2 and maximum degree at most 3 is a graph in which every vertex has degree at least 2 and at most 3.""",
    "a connected and chordal graph with min_degree at least 2": r"""A connected and chordal graph with minimum degree at least 2 is a graph in which every vertex has degree at least 2 and every induced cycle has exactly three vertices.""",
    "a connected and chordal graph with min_degree at least 3": r"""A connected and chordal graph with minimum degree at least 3 is a graph in which every vertex has degree at least 3 and every induced cycle has exactly three vertices.""",
    "a connected and chordal graph with min_degree at least 4": r"""A connected and chordal graph with minimum degree at least 4 is a graph in which every vertex has degree at least 4 and every induced cycle has exactly three vertices.""",
    "a connected and bull-free graph": r"""A *bull* is the complete graph $K_4$ minus one edge. A *bull-free graph* is a graph in which no induced subgraph is a bull.""",
    "a connected and diamond-free graph": r"""A *diamond* is a graph formed by removing one edge from the complete graph $K_4$. A *diamond-free graph* is a graph in which no induced subgraph is a diamond.""",
    "a connected, cubic, and diamond-free graph": r"""A *cubic graph* is a graph where every vertex has degree 3. A *diamond* is a graph formed by removing one edge from the complete graph $K_4$. A *diamond-free graph* is a graph in which no induced subgraph is a diamond.""",
    "a block graph": """A block graph is a connected graph in which every biconnected component is a clique.""",


}

TRIVIAL_BOUNDS = [
    "domination_number <= order",
    "domination_number <= roman_domination_number",
    "domination_number <= double_roman_domination_number",
    "domination_number <= two_rainbow_domination_number",
    "domination_number <= three_rainbow_domination_number",
    "order <= size - 1",
    "domination_number <= total_domination_number",
    "domination_number <= independent_domination_number",
    "domination_number <= connected_domination_number",
    "domination_number <= independence_number",
    "domination_number <= 2 min_maximal_matching_number",
    "domination_number >= 1/3 double_roman_domination_number",
    "independent_domination_number >= 1/3 double_roman_domination_number",
    "total_domination_number >= 1/3 double_roman_domination_number",
    "connected_domination_number >= 1/3 double_roman_domination_number",
    "independence_number >= 1/3 double_roman_domination_number",
     "independence_number >= 1/2 roman_domination_number",
    "restrained_domination_number >= domination_number",
    "domination_number <= restrained_domination_number",
    "total_domination_number <= 2 min_maximal_matching_number",
    "two_rainbow_domination_number >= domination_number",
    "three_rainbow_domination_number >= domination_number",
    "three_rainbow_domination_number >= two_rainbow_domination_number",
    "domination_number >= power_domination_number",
    "power_domination_number <= zero_forcing_number",
    "power_domination_number <= domination_number",
    "power_domination_number <= total_domination_number",
    "power_domination_number <= connected_domination_number",
    "power_domination_number <= total_zero_forcing_number",
    "independent_domination_number <= independence_number",
    "total_domination_number <= connected_domination_number",
    "domination_number <= outer_connected_domination_number",
    "outer_connected_domination_number >= domination_number",
    "outer_connected_domination_number >= slater",
    "roman_domination_number >= domination_number",
    "double_roman_domination_number >= roman_domination_number",
    "roman_domination_number >= slater",
    "double_roman_domination_number >= slater",
    "roman_domination_number <= double_roman_domination_number",
    "roman_domination_number <= 2 domination_number",
    "roman_domination_number <= 2 independent_domination_number",
    "roman_domination_number <= 2 total_domination_number",
    "roman_domination_number <= 2 connected_domination_number",
    "double_roman_domination_number <= 3 domination_number",
    "double_roman_domination_number <= 3 independent_domination_number",
    "double_roman_domination_number <= 3 total_domination_number",
    "double_roman_domination_number <= 3 connected_domination_number",
    "double_roman_domination_number >= 2 domination_number",
    "zero_forcing_number >= min_degree",
    "positive_semidefinite_zero_forcing_number <= zero_forcing_number",
    "positive_semidefinite_zero_forcing_number <= total_zero_forcing_number",
    "positive_semidefinite_zero_forcing_number <= connected_zero_forcing_number",
    "zero_forcing_number >= positive_semidefinite_zero_forcing_number",
    "zero_forcing_number <= total_zero_forcing_number",
    "total_zero_forcing_number >= zero_forcing_number",
    "total_zero_forcing_number >= positive_semidefinite_zero_forcing_number",
    "power_domination_number >= total_zero_forcing_number",
    "zero_forcing_number >= power_domination_number",
    "zero_forcing_number <= (order - connected_domination_number)",
    "total_zero_forcing_number >= min_degree",
    "positive_semidefinite_zero_forcing_number >= min_degree",
    "matching_number >= min_maximal_matching_number",
    "min_maximal_matching_number <= matching_number",
    "matching_number <= 1/2 order",
    "connected_domination_number >= domination_number",
    "total_domination_number >= domination_number",
    "total_domination_number >= sub_total_domination_number",
    "domination_number >= 1/2 total_domination_number",
    "domination_number >= 1/2 roman_domination_number",
    "independent_domination_number >= 1/2 roman_domination_number",
    "total_domination_number >= 1/2 roman_domination_number",
    "connected_domination_number >= 1/2 roman_domination_number",
    "domination_number <= matching_number",
    "domination_number <= vertex_cover_number",
    "domination_number <= 1/2 order",
    "domination_number <= (order - max_degree)",
    "domination_number >= slater",
    "edge_domination_number >= LG_slater",
    "total_domination_number <= 2 domination_number",
    "total_domination_number <= 2 independent_domination_number",
    "power_domination_number <= independent_domination_number",
    "total_zero_forcing_number <= order + -1",
    "zero_forcing_number <= order + -1",
    "total_domination_number <= order + -1",
    "domination_number <= order + -1",
    "independent_domination_number <= order + -1",
    "connected_domination_number <= order + -1",
    "power_domination_number <= order + -1",
    "min_degree <= max_degree",
    "min_degree <= zero_forcing_number",
    "annihilation_number >= matching_number",
    "annihilation_number >= independence_number",
    "annihilation_number >= residue",
    "independence_number <= annihilation_number",
    "independence_number >= radius",
    "sub_total_domination_number <= total_domination_number",
    "sub_total_domination_number >= slater",
    "slater <= domination_number",
    "chromatic_number >= clique_number",
    "independent_domination_number >= domination_number",
    "matching_number <= (order - matching_number)",
    "independent_domination_number <= (order - max_degree)",
    "independent_domination_number >= slater",
    "independent_domination_number >= power_domination_number",
    "independent_domination_number >= 1/2 total_domination_number",
    "independent_domination_number <= 1/2 order",
    "independent_domination_number >= [order/ (max_degree + 1)]",
    "independent_domination_number <= 1/2 power_domination_number + 11/2",
    "independent_domination_number <= 2/3 zero_forcing_number + 13/3",
    "independent_domination_number <= 1/5 diameter + 27/5",
    "domination_number >= [order/ (max_degree + 1)]",
    "connected_domination_number >= power_domination_number",
    "independence_number <= radius",
    "independence_number >= residue",
    "independence_number <= order + -1",
    "independence_number <= size",
    "independence_number <= (order - matching_number)",
    "independence_number >= independent_domination_number",
    "independence_number >= domination_number",
    "diameter >= radius",
    "zero_forcing_number >= 1/2 total_zero_forcing_number",
    "zero_forcing_number <= connected_zero_forcing_number",
    "annihilation_number <= 1/2 order",
    "connected_zero_forcing_number >= zero_forcing_number",
    "connected_zero_forcing_number <= zero_forcing_number",
    "connected_zero_forcing_number >= min_degree",
    "connected_zero_forcing_number <= order + -1",
    "connected_zero_forcing_number >= chromatic_number + -1",
    "connected_zero_forcing_number >= clique_number + -1",
    "connected_zero_forcing_number >= total_zero_forcing_number",
    "connected_zero_forcing_number >= positive_semidefinite_zero_forcing_number",
    "zero_forcing_number >= chromatic_number + -1",
    "zero_forcing_number >= clique_number + -1",
    "total_zero_forcing_number >= chromatic_number + -1",
    "total_zero_forcing_number >= clique_number + -1"
    "connected_zero_forcing_number >= chromatic_number + -1",
    "connected_zero_forcing_number >= clique_number + -1",
    "slater <= independent_domination_number",
    "residue <= independence_number",
    "[order/ (max_degree + 1)] <= domination_number",
    "annihilation_number >= (order - matching_number)",
    "annihilation_number >= domination_number",
    "annihilation_number >= independence_number",
    "annihilation_number >= residue",
    "annihilation_number >= matching_number",
    "annihilation_number >= independent_domination_number",
    "annihilation_number >= slater",
    "annihilation_number >= power_domination_number",
    "double_roman_domination_number <= 3 restrained_domination_number",
    "double_roman_domination_number <= 3 outer_connected_domination_number",
    "double_roman_domination_number <= 2 two_rainbow_domination_number",
    "restrained_domination_number >= slater",
    "restrained_domination_number >= 1/2 roman_domination_number",
    "restrained_domination_number >= power_domination_number,",
    "restrained_domination_number >= 1/3 double_roman_domination_number",
    "restrained_domination_number >= power_domination_number",
    "triameter <= 3 diameter",
    "slater <= sub_total_domination_number",
    "slater <= restrained_domination_number",
    "slater <= outer_connected_domination_number",
    "slater <= total_domination_number",
    "slater <= connected_domination_number",
    "domination_number <= 1/2 double_roman_domination_number",
    "total_domination_number <= 2 edge_domination_number",
    "power_domination_number <= 1/2 total_zero_forcing_number",
    "zero_forcing_number <= size + -1",
    "total_zero_forcing_number >= 2 power_domination_number",
    "total_zero_forcing_number <= size",
    "diameter <= 2 radius",
    "radius <= diameter",
    "radius >= 1/2 diameter",
    "radius <= independence_number",
    "independent_domination_number <= (order - independent_domination_number)",
    "independent_domination_number <= (order - domination_number)",
    "chromatic_number <= zero_forcing_number + 1",
    "chromatic_number <= clique_number + 1",
    "chromatic_number <= max_degree + 1",
    "clique_number <= chromatic_number",
    "clique_number <= zero_forcing_number + 1",
    "clique_number <= max_degree + 1",
    "residue <= power_min_degree_residue_sum",
    "residue <= annihilation_number",
    "annihilation_number <= power_min_degree_annihilation_sum",
    "(order - domination_number) >= (order - independent_domination_number)",
    "(order - domination_number) <= (order - slater)",
    "(order - domination_number) >= (order - total_domination_number)",
    "(order - domination_number) <= (order - power_domination_number)",
    "(order - domination_number) >= domination_number",
    "(order - domination_number) <= order + -1",
    "(order - domination_number) >= vertex_cover_number",
    "(order - domination_number) >= (order - connected_domination_number)",
    "(order - total_domination_number) <= (order - sub_total_domination_number)",
    "(order - total_domination_number) <= (order - domination_number)",
    "(order - connected_domination_number) <= (order - domination_number)",
    "(order - connected_domination_number) <= order + -1",
    "(order - power_domination_number) <= order + -1",
    "(order - power_domination_number) >= (order - domination_number)",
    "(order - zero_forcing_number) <= (order - min_degree)",
    "(order - zero_forcing_number) <= (order - chromatic_number) + 1",
    "(order - zero_forcing_number) <= order + -1",
    "(order - zero_forcing_number) <= (order - power_domination_number)",
    "min_edge_cover <= order + -1",
    "positive_semidefinite_zero_forcing_number <= vertex_cover_number",



]

import time

def long_computation():
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f'Filtering and sorting the proposed inequalities... {i + 1}%')
        time.sleep(0.1)  # Simulate a long computation

    status_text.text('Done!')
    st.success('Process complete!')


def fraction_to_str(fraction):
    return f"{fraction.numerator}/{fraction.denominator}"

def str_to_fraction(string):
    numerator, denominator = map(int, string.split('/'))
    return Fraction(numerator, denominator)

def conjecture_to_dict(conjecture):
    return {
        "hypothesis": conjecture.hypothesis.statement,
        "conclusion": {
            "lhs": conjecture.conclusion.lhs[0],
            "inequality": conjecture.conclusion.inequality,
            "slope": fraction_to_str(conjecture.conclusion.slope) if isinstance(conjecture.conclusion.slope, Fraction) else conjecture.conclusion.slope,
            "rhs": conjecture.conclusion.rhs,
            "intercept": fraction_to_str(conjecture.conclusion.intercept) if isinstance(conjecture.conclusion.intercept, Fraction) else conjecture.conclusion.intercept
        },
        "symbol": conjecture.symbol,
        "touch": conjecture.touch
    }

def conjecture_to_latex(conjecture):
    conj_dict = conjecture_to_dict(conjecture)
    slope_numerator = conjecture.conclusion.slope.numerator
    slope_denominator = conjecture.conclusion.slope.denominator
    if slope_numerator == slope_denominator:
        slope = ""
    else:
        slope = r"\frac{" + f"{slope_numerator}" + "}{" + f"{slope_denominator}" + "}" if slope_denominator != 1 else str(slope_numerator)

    intercept_numerator = abs(conjecture.conclusion.intercept.numerator)
    intercept_denominator = abs(conjecture.conclusion.intercept.denominator)
    operation = "+" if conjecture.conclusion.intercept.numerator >= 0 else "-"
    if intercept_numerator == 0:
        tex_string = TEX_MAP[conj_dict["conclusion"]["lhs"]] + " " + TEX_MAP[conj_dict["conclusion"]["inequality"]] + " " + slope + "" + TEX_MAP[conj_dict["conclusion"]["rhs"]] + ","
    elif intercept_numerator == 1 and intercept_denominator == 1:
        tex_string = TEX_MAP[conj_dict["conclusion"]["lhs"]] + " " + TEX_MAP[conj_dict["conclusion"]["inequality"]] + " " + slope + "" + TEX_MAP[conj_dict["conclusion"]["rhs"]] + f" {operation} " + "1"+ ","
    else:
        intercept = r"\frac{" + f"{intercept_numerator}" + "}{" + f"{intercept_denominator}" + "}" if intercept_denominator != 1 else str(intercept_numerator)
        tex_string = TEX_MAP[conj_dict["conclusion"]["lhs"]] + " " + TEX_MAP[conj_dict["conclusion"]["inequality"]] + " " + slope + "" + TEX_MAP[conj_dict["conclusion"]["rhs"]] + f" {operation} " + intercept + ","
    return tex_string

def multi_radio(label, options):
    selections = {option: False for option in options}

    st.write(label)
    for option in options:
        selections[option] = st.checkbox(option, key=option)

    selected_options = [option for option, selected in selections.items() if selected]
    return selected_options

def rows_multi_radio(label, options):
    st.write(label)
    selections = {option: False for option in options}
    rows = (len(options) + 3) // 2  # Calculate number of rows needed

    for i in range(rows):
        cols = st.columns(2)
        for j in range(2):
            if 2 * i + j < len(options):
                option = options[2 * i + j]
                selections[option] = cols[j].checkbox(option, key=option)

    selected_options = [option for option, selected in selections.items() if selected]
    return selected_options

def generate_conjectures():
    # st.title("Generate Conjectures")
    st.set_page_config(page_title="Conjecture Generator") #, page_icon="📈")
    st.markdown("# Conjecturing with TxGraffiti on Graph Products")
    # st.sidebar.header("Plotting Demo")
    st.markdown("## Notation:")
    st.write(r"""$G \mathbin{\square} H$ denotes the *Cartesian product*.""")
    st.write(r"""$G \times H$ denotes the *tensor product*.""")
    st.write(r"""$G \circ H$ denotes the *lexicographic product*.""")
    st.write(r"""$G \boxtimes H$ denotes the *strong product*.""")

    st.markdown("## Instructions:")
    st.write(
        r"""Please select one or more categories to conjecture on below. After selecting your categories, generate conjectures
        by clicking the 'Generate Conjectures' button. The conjectures will be computed and then displayed below. NOTE: This process may take a few minutes.

        """
    )



    df = pd.read_csv(DATA_FILE)


    #set the index to the name of the graph
    df.set_index("name", inplace=True)

    # print(df.cartesian_product_domination_number)
    # invariants = list(df.columns) - ["name"]
    invariants = [x for x in df.columns if x not in ["name", "is_connected"]]
    callables = [x for x in invariants if "product" in x]
    # print(callables)
    # target = "cartesian_product_domination_number"
    # columns = [x for x in invariants if x not in ["name"] and "cartesian" not in x]


    # data = st.button("Update Graph Database")
    # if data:
    #     make_graph_dataframe_from_edgelists()

    # with st.sidebar:

    invariant_column = rows_multi_radio('### Select categories to conjecture on (the more you select the longer TxGraffiti will take to learn):', callables)

    if invariant_column:
        if "cartesian" in invariant_column[0]:
            invariants = [x for x in invariants if "cartesian" not in x]
        elif "lexicographic" in invariant_column[0]:
            invariants = [x for x in invariants if "lexicographic" not in x]
        elif "strong" in invariant_column[0]:
            invariants = [x for x in invariants if "strong" not in x]
        elif "tensor" in invariant_column[0]:
            invariants = [x for x in invariants if "tensor" not in x]
    # removal_invariants = multi_radio('### Exclude any invariants?', numerical_columns)
    # invariant_column = [invariant for invariant in invariant_column if invariant not in removal_invariants]



    generate_conjectures = st.button('Generate Conjectures')
    conjectures = []
    if generate_conjectures:
        for invariant in invariant_column:
            with st.spinner(f'Learning conjectures for the {invariant} ...'):
                upper_conjectures = make_all_upper_linear_conjectures(df, invariant, invariants, ["is_connected"])
                lower_conjectures = make_all_lower_linear_conjectures(df, invariant, invariants, ["is_connected"])
                conjectures = conjectures + upper_conjectures + lower_conjectures
                long_computation()


        conjs = [conj for conj in conjectures if conj.touch > 0]
        conjs.sort(key=lambda x: x.touch, reverse=True)
        conjs = dalmatian(df, conjs)
        for conj in conjs:
            print(conj.conclusion)
        # conjectures = dalmatian(df, conjectures)


        st.subheader("TxGraffiti conjectures the following inequalities:")
        for i, conjecture in enumerate(conjs):
            with st.expander(f"# Conjecture {i + 1}"):
                lhs = TEX_MAP[conjecture.conclusion.lhs]
                rhs = TEX_MAP[conjecture.conclusion.rhs]
                inequality = TEX_MAP[conjecture.conclusion.inequality]
                if conjecture.conclusion.slope == 1:
                    slope = ""
                elif conjecture.conclusion.slope.denominator == 1:
                    slope = str(conjecture.conclusion.slope.numerator)
                else:
                    slope = fraction_to_str(conjecture.conclusion.slope)
                # slope = fraction_to_str(conjecture.conclusion.slope) if conjecture.conclusion.slope != 1 else ""
                intercept = fraction_to_str(conjecture.conclusion.intercept) if conjecture.conclusion.intercept != 0 else ""

                if conjecture.conclusion.intercept > 0:
                    if conjecture.conclusion.intercept.denominator == 1:
                        st.write(f"If $G$ and $H$ are nontrivial connected graphs, then ")
                        st.write(f"$${lhs} {inequality} {slope} {rhs} + {conjecture.conclusion.intercept.numerator}$$,")
                        # st.write(f"and this bound is sharp.")
                    else:
                        st.write(f"If $G$ and $H$ are nontrivial connected graphs, then ")
                        st.write(f"$${lhs} {inequality} {slope} {rhs} + {intercept}$$,")
                        # st.write(f"and this bound is sharp.")
                elif conjecture.conclusion.intercept < 0:
                    if conjecture.conclusion.intercept.denominator == 1:
                        st.write(f"If $G$ and $H$ are nontrivial connected graphs, then")
                        st.write(f"$${lhs} {inequality} {slope} {rhs} {conjecture.conclusion.intercept.numerator}$$,")
                        # st.write(f"and this bound is sharp.")
                    else:
                        st.write(f"If $G$ and $H$ are nontrivial connected graphs, then ")
                        st.write(f"$${lhs} {inequality} {slope} {rhs} {intercept}$$,")
                        # st.write(f"and this bound is sharp.")
                else:
                    st.write(f"If $G$ and $H$ are nontrivial connected graphs, then ")
                    st.write(f"$${lhs} {inequality} {slope} {rhs}$$,")

                st.write(f"and this bound is sharp on {conjecture.touch} graphs.")

# Main entry point
if __name__ == "__main__":
    generate_conjectures()
