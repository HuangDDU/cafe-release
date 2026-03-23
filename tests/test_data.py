import cafe


def get_test_fadata():
    fadata = cafe.data.read_dynverse_simulation_data()

    # rename milestones
    milestone_old2new = {
        "M1": "A",
        "M3": "B",
        "M4": "C",
        "M2": "D",
    }
    mw = fadata.get_milestone_wrapper()
    mw.rename_milestone(milestone_old2new)

    # cluster: group cells onto nearest milestones
    cluster = "group_onto_ref"
    fadata.group_onto_nearest_milestones(cluster_key=cluster)

    # milestone realated color
    fadata.obs[cluster] = fadata.obs[cluster].astype("category")
    fadata.uns[f"{cluster}_colors"] = list(map(lambda x: mw.milestone_color_dict[x], fadata.obs[cluster].cat.categories))

    # basis = "X_dynverse"
    # fadata.obsm[basis] = pd.read_csv("tmp/bifurcating_1_dimred.csv", index_col=0).values
    # basis

    prior_information = {
        "cluster": cluster,
        # "basis": basis,
        "start_cell": "C445",
        "start_milestone": "A",
    }
    fadata.add_prior_information(**prior_information)

    return fadata
