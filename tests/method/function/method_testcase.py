def method_testcase(adata, method_name, backend, parameters):
    import cfe

    fadata = cfe.data.FateAnnData.from_anndata(adata)
    method = cfe.method.FateMethod(method_name, backend_name=backend)
    method.infer_trajectory(fadata, parameters)

    return fadata
