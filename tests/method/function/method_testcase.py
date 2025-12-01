def method_testcase(adata, method_name, backend, parameters):
    import cafe

    fadata = cafe.data.FateAnnData.from_anndata(adata)
    method = cafe.method.FateMethod(method_name, backend_name=backend)
    method.infer_trajectory(fadata, parameters)

    return fadata
