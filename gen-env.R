rix::rix(
  date = "2026-01-05",
  r_pkgs = NULL,
  system_pkgs = NULL,
  tex_pkgs = NULL,

  py_conf = list(
    py_version = "3.12",
    py_pkgs = c(
      "numpy", "pandas",
      "matplotlib", "seaborn",
      "scikit-learn", "statsmodels",
      "openpyxl",
      "pytest",
      "jinja2",
      "weasyprint"
    )
  ),

  ide = "none",

  project_path = ".",
  overwrite = TRUE
)
