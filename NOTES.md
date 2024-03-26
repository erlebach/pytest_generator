- I will create four Yaml files, one per test. 
- New types: lineplots, scattergram
  scattergram: dictionary. 
  - key: color
  - values: (float1, float2)

Part 1, evaluation of k-means over diverse datasets

1A: 

- Load the datasets. I will not check this. 

1B: 

- Check existence of function 
   fit_kmeans(X: NDArray, label: NDArray, nb_clusters: int) -> predicted labels from K-means 
   - standardize the data
   - fit the data (ONLY CHECK THE FIT)

  TEST: check that the function runs with a simple dataset

1C: 

20 scatterplots. Upload the pdf of these figures. 
Rows == nb clusters: k=[2,3,5,10]

Datasets: A: "noisy_circles", B: "noisy_moons", C: "blobs with varied variances", D: Anisotropicly distributed data", E: blobs
answer["1C, failed"] = ["A", "B", "C", D"]
answer["1C, success"] = [("E",), (k1, k2)]  : first element of the list: a sequence of datasets (add comma for a sequence of 1)
                                            : second element of the list: a sequence of cluster numbers for which the clusters were correct

1D: 

repeat 1C a five times with different initializations. No plots. 
answer["1D, sensitive datasets"]: list of datasets in ["A" through "E"]

----------------------------------------------------------------------
Part 2, Comparison of Clustering Evaluation Metrics
Save as part 1A and part 1B with hierarchical clustering. 

2A: 

call make_blobs (this can be checked)

2B: 

modify fit_kmeans to return SSE

2C: 

Plot SSE as function of k = 1,2,...,8. 

answer["2C: SSE plot"] = dict[k, list of lists of x,y]
answer["2C: optimal k"] = int   (I expect the answer to be correct to within +- 1

2D:  Samve as 2C using inertia. 
answer["2D: intertia plot"] = dict[k, list of lists of x,y]
answer["2D: optimal k"] = int   (I expect the answer to be correct to within +- 1

----------------------------------------------------------------------
Part 3, Hierarchical Clustering
  def fit_hier_cluster(dataset, linkage type, k: int) -> predictions

3A. Load hierarchical_toy_data.mat  (no testing)

3B. Linkage Matrix Z: output this matrix. 
Test: calculate the matrix mean and standard deviation (compare with student). 'single' linmkage

answer["3B: matrix"]: matrix. 

3C.

answer["3C: iteration merge"]

3D.
answer["3D: dissimilarity"]: float   
Check: same value as the 3rd column of the row in problem 2C.  (the students should not get this hint)
I can use this check in the testing. 

3E. Answer["3E: clusters"]: [[2,3,5], [6,2,3]]
In the test: transform lists to sets. And then check. 
Return: {6,14}, {4}, {5}, {11}, {0}, {10}, {3}, {7}, and {12}.

>>> Somehow ,return the dendogram so I can test it. SKIP FOR NOW. 

3F. SKIP. 

Answer["3F: produce two rings"]:   "yes" or "no"
Answer["3F: produce two rings, describe"]:   "string"

3G: Answer["3G:     ] 



----------------------------------------------------------------------
Part 4, Evaluation of Hierarchical Clustering over Diverse Datasets

Repeat part 1.AB with hierarchical clustering. Return the plot with 4 linkage (single, complete, ward, centroid)
    2 clusters for all runs. 

List the data sets that are correclty clusters that k-means did not. 

answer["4B: datasets"]: ["A", "B"]  ("circles and "Moons")


4C: 

answer["4C: cut-off distance"]
answer["4C: datasets"]  (same question as 4B)
   correct_answer: never get the correct clustering. 
   NOT CLEAR WHETHER I SHOULD KEEP THIS. 

----------------------------------------------------------------------
# NOTES: 
- All plotting functions should be in a file "myplots.py".
"import myplots" in your code. Use any library you wish. 
----------------------------------------------------------------------
# 2024-03-14
Wrote 
``` ./generate_debug_tests.x test1 ```
which generates the file `preprocessed_test1_expand.yaml` for input into test_answers_generator.py and test_structure_generator.py. 
----------------------------------------------------------------------
# 2024-03-16
Structural tests pass and fail correctly with test4.yaml and test5.yaml. The explanation method is listed. 
TODO: test with Gradescope with fake homework. 
TODO next: work on generate*answ*gen*py
All structure and answer tests pass with messages
TODO: scoring
----------------------------------------------------------------------
# 2024-03-17
It might be possible to have only a single generator, but use options in generator_config.py to decide what to creates. For now leave as is. 
But the fact is that the answer generator and structure generator are close. 
Since the answer test  first runs the structure test, I should be able to control the flow within the test itself. 
----------------------------------------------------------------------
2024-03-20
- The answers are correct.  (test_answer...) although I insert errors in the py files. Something not right. 
- running the generator updates the files in tests/  (I checked htis)
- type:int is treated properly
  type:integer is not.o Yet they are both defined the same way in assert_utilities. BUT without resolution.
----------------------------------------------------------------------
2024-03-21_22:35
TODO: 
  - I don't think there is a need for a generator to create a generator. At least not for the time being. I have not yet found a real use. 
  - I should also consolidate the structure and answer generator. Thus, I could add an argument --type 'answer', or --type 'structure' to generate structure or answer tests. 
  - To create more composite types, I could compose the functions in assert_utilities. 
  - Add a LLM component to these tests, which can be optionally turned on or off. 
----------------------------------------------------------------------
2024-03-22_11:40
- I now only have test_generator.py
- The code is greatly simplified. 
- ISSUE: structure: explanation is null (should be easily fixable). 
- Think of how to add LLM. 
----------------------------------------------------------------------
2024-03-25_15:38
Pydantic validates function arguments of many different types and is customizable. 
----------------------------------------------------------------------
