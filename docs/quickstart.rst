Quick Start
===========

This guide walks through a typical analysis workflow using ``costim-screen``.

Loading Data
------------

Start by importing the package and loading your data files:

.. code-block:: python

   import costim_screen as cs
   from pathlib import Path

   # Define paths
   data_path = Path("data")
   results_path = Path("results")
   results_path.mkdir(parents=True, exist_ok=True)

   # Load data
   counts = cs.load_counts_matrix(data_path / "merged_counts.xlsx")
   smeta = cs.load_sample_metadata(data_path / "sample_metadata.xlsx")
   cand = cs.load_candidate_metadata(data_path / "candidate_metadata.xlsx")


Preprocessing
-------------

Filter low-abundance candidates and prepare the data:

.. code-block:: python

   # Filter candidates with low total counts
   counts = cs.filter_domains_by_total_counts(counts, min_total=50)
   cand = cand.loc[counts.index]


Building the ELM Design Matrix
------------------------------

Create a one-hot encoded design matrix from ELM annotations:

.. code-block:: python

   X_elm = cs.build_elm_category_design(
       cand.reset_index(),
       candidate_id_col="CandidateID",
       elm_col="ELMCategory",
       min_freq=0.025,
   )

   # Make column names patsy-safe
   safe_cols, mapping = cs.make_patsy_safe_columns(list(X_elm.columns), prefix="ELM_")
   X_elm = X_elm.rename(columns=mapping)


Preparing Long-Format Data
--------------------------

Convert to long format and add sample-level variables:

.. code-block:: python

   # Add derived variables
   smeta["phenotype"] = smeta["Tsubset"] + "_" + smeta["PD1Status"]
   smeta["CCR"] = cs.make_ccr_id(smeta)

   # Convert to long format
   df = cs.counts_to_long(counts, id_col="CandidateID")
   df = df.merge(smeta.reset_index(), on="sample_id", how="left")
   df = cs.add_library_size(df)

   # Attach ELM features
   df = df.merge(
       X_elm.reset_index().rename(columns={"index": "CandidateID"}),
       on="CandidateID",
       how="left"
   )


Fitting the Model
-----------------

Fit a negative binomial GLM with motif-phenotype interactions:

.. code-block:: python

   # Subset to condition of interest
   df_raji = df[df["ExpCond"] == "CAR:Raji"].copy()
   df_raji["phenotype"] = pd.Categorical(df_raji["phenotype"])

   # Build formula and fit
   motif_cols = list(X_elm.columns)
   formula = cs.build_joint_formula(motif_cols)
   fit = cs.fit_nb_glm_iter_alpha(
       df_raji,
       formula=formula,
       offset_col="offset",
       cluster_col="CCR"
   )

   print(f"Estimated dispersion: {fit.alpha:.3f}")


Computing Contrasts
-------------------

Test for differential motif effects between phenotypes:

.. code-block:: python

   # Compare EM_High vs CM_High
   result = cs.motif_contrast_table(
       fit,
       motifs=motif_cols,
       p="EM_High",
       q="CM_High",
       adjust="BH",
       log_base=2.0,
   )

   # View significant results
   print(result[result["qvalue"] < 0.10])


Generating Volcano Plots
------------------------

Visualize the results:

.. code-block:: python

   cs.volcano_plot(
       result,
       q_thresh=0.10,
       lfc_thresh=1.0,
       title="EM_High vs CM_High",
       outpath=results_path / "volcano.png"
   )


Pooled Contrasts
----------------

For comparing groups while pooling over other factors:

.. code-block:: python

   # T-subset comparison pooled over PD1
   tab_em_vs_cm = cs.volcano_tsubset_pooled_pd1(
       fit,
       motif_cols,
       tsubset_p="EM",
       tsubset_q="CM",
       q_thresh=0.10,
       lfc_thresh=1.0,
       outpath=results_path / "volcano_EM_vs_CM_pooled.png"
   )

   # PD1 comparison pooled over T-subsets
   tab_pd1 = cs.volcano_pd1_pooled_tsubset(
       fit,
       motif_cols,
       tsubsets=("Naïve", "CM", "EM"),
       pd1_high="High",
       pd1_low="Low",
       outpath=results_path / "volcano_PD1_pooled.png"
   )