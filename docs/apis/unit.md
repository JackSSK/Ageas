### [Tutorial](https://nkmtmsys.github.io/Ageas/tutorial)


# Unit()
Object containing AGEAS extraction unit.

> ageas.Unit(database_info = None, meta = None, model_config = None, pseudo_grns = None, clf_keep_ratio:float = 0.5, clf_accuracy_thread:float = 0.8, correlation_thread:float = 0.2, cpu_mode:bool = False, feature_dropout_ratio:float = 0.1, feature_select_iteration:int = 1, grp_changing_thread:float = 0.05, impact_depth:int = 3, link_step_allowrance:int = 0, max_train_size:float = 0.95, model_select_iteration:int = 2, outlier_thread:float = 3.0, stabilize_patient:int = 3, stabilize_iteration:int = 10, top_grp_amount:int = 100, z_score_extract_thread:float = 0.0,)


## **Args**

+ **_database_info_**: Default = None

    Integrated database information returned by ageas.Get_Pseudo_Samples()


+ **_meta_**: Default = None

    meta level processed grand GRN information returned by ageas.Get_Pseudo_Samples()


+ **_model_config_**: Default = None




+ **_pseudo_grns_**: Default = None

    pseudo-sample GRNs returned by ageas.Get_Pseudo_Samples()


+ **_clf_keep_ratio_**: Default = 0.5

    Portion of classifier model to keep after each model selection iteration.
    + Note: When performing SHA based model selection, this value is set as lower bound to keep models

+ **_clf_accuracy_thread_**: Default = 0.8

    Filter thread of classifier's accuracy in local test performed at each model selection iteration.
    + Note: When performing SHA based model selection, this value is only used at last iteration.


+ **_correlation_thread_**: Default = 0.2

    Gene expression correlation thread value of GRPs.

    Potential GRPs failed to reach this value will be dropped.


+ **_cpu_mode_**: Default = False

    Whether force to use CPU only or not.

+ **_feature_dropout_ratio_**: Default = 0.1

    Portion of features(GRPs) to be dropped out after each iteration of feature selection.

+ **_feature_select_iteration_**: Default = 1

    Number of iteration for feature(GRP) selection before key GRP extraction.

+ **_grp_changing_thread_**: Default = 0.05

    If changing portion of key GRPs extracted by AGEAS unit from two stabilize iterations lower than this thread, these two iterations will be considered as having consistent result.

+ **_impact_depth_**: Default = 3

    When assessing a TF's regulatory impact on other genes, how far the distance between TF and potential regulatory source can be.
    + Note: The correlation strength of stepped correlation strength of TF and gene still need to be greater than correlation_thread.

+ **_link_step_allowrance_**: Default = '1

    During key atlas extraction, when finding bridge GRPs to link 2 separate regulons, how many steps will be allowed.
    + e.g.: link_step_allowrance == 1 means, no intermediate gene can be used and portential regulatory source must be able to interact with gene from another regulon.

+ **_max_train_size_**: Default = 0.95

    The largest portion of avaliable data can be used to train models.

    At the mandatory model filter, this portion of data will be given to each model to train.

+ **_model_select_iteration_**: Default = 2

    Number of iteration for classification model selection before the mandatory filter.

+ **_outlier_thread_**: Default = 3.0

    The lower bound of Z-score scaled importance value to consider a GRP as outlier need to be retain.

+ **_stabilize_patient_**: Default = 3

    If stabilize iterations continuously having consistent result for this value, an early stop on result stabilization will be executed.

+ **_stabilize_iteration_**: Default = 10

    Number of iteration for a AGEAS unit to repeat key GRP extraction after model and feature selections in order to find key GRPs consistently being important.

+ **_top_grp_amount_**: Default = 100

    Amount of GRPs an AGEAS unit would extract.
    + Note: If outlier_thread is set, since outlier GRPs are extracted during feature selection part and will also be considered as key GRPs, actual amount of key GRPs would be greater.

+ **_z_score_extract_thread_**: Default = 0.0

    The lower bound of Z-score scaled importance value to extract a GRP.


## **Methods**


+ **save_reports()**

  Save data and reports in given folder

  + **Args**

    + **_folder_path_**: Default = None

        Folder path to save all files.


    + **_save_unit_reports_**: Default = False

        Whether save key GRPs extracted by each AGEASS


## **Attributes**



## **Example**
```python
import ageas
result = ageas.Launch(
  group1_path = 'test/ips.csv',
  group2_path = 'test/mef.csv',
)
result.save_reports(
	folder_path = 'easy_test.report',
	save_unit_reports = True
)
```
