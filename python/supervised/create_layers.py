
##Create layers
#activation = ('relu', 'tanh', 'sigmoid')
#units = (800, 700, 600, 500, 400, 300)
#p = list(product(activation, units))
#c_1 = list(map(list, list(combinations(p, 1))))
#c_2 = list(map(list, list(combinations(p, 2))))
#c_1.extend(c_2)
#k = []
#for e in c_1:
#    k.append(sorted(e, reverse=True, key=lambda x: x[1]))
#unique = [list(x) for x in set(tuple(x) for x in k)]
#for combo in unique:
#    combo.append(('softmax', len(crystal_labels)))
#all_model_layers = unique

    ##Testdata dirs
    #quasidata_dir = '/home/kristtuv/Documents/master/src/python/allquasidata/'
    #quasidata_adjacency_dir = 'quasicrystal_files_adjacency/'
    #methanedata_dir = '/home/kristtuv/Documents/master/src/python/Grace_ase/datafiles/' 
    #methanedata_adjacency_dir = '/home/kristtuv/Documents/master/src/python/Grace_ase/dumpfiles/adjacency/' 
    #results_dumpdir = 'results/'
    #model_evaluation_dumpdir = 'model_evaluation/'
    #model_dumpdir = 'nn_models/'

    ## all_model_layers = [[('relu', 800),('relu', 600),('relu', 300),  ('softmax', len(crystal_labels))]]
    #adjacency_files, _ = make_file(quasidata_adjacency_dir, model_dumpdir)
    #data_files, dump_names = make_file(quasidata_dir, results_dumpdir, exclude='.log')
    #adjacency_files, data_files, dump_names = sorted(adjacency_files), sorted(data_files), sorted(dump_names)

    ##Create layers
    #activation = ('relu', 'tanh', 'sigmoid')
    #units = (800, 700, 600, 500, 400, 300)
    #p = list(product(activation, units))
    #c_1 = list(map(list, list(combinations(p, 1))))
    #c_2 = list(map(list, list(combinations(p, 2))))
    #c_1.extend(c_2)
    #k = []
    #for e in c_1:
    #    k.append(sorted(e, reverse=True, key=lambda x: x[1]))
    #unique = [list(x) for x in set(tuple(x) for x in k)]
    #for combo in unique:
    #    combo.append(('softmax', len(crystal_labels)))
    #all_model_layers = unique

    #run_predict = False
    #recompute_model = False
    #dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    #batch_sizes = [512, 256]
    #best_model = None
    #best_evaluation = [1000000, 0]
    #for model_layers in all_model_layers:
    #    for batch_size in batch_sizes:
    #        for dropout_rate in dropout_rates:
    #            layer_names = '_'.join(map(str, model_layers))
    #            layer_names = layer_names.replace("'", '')
    #            layer_names = layer_names.replace(" ", '')
    #            layer_names = layer_names.replace("(", '')
    #            layer_names = layer_names.replace(")", '')
    #            layer_names = layer_names.replace(",", '')
    #            epochs = 50
    #            model_dumpname = (
    #                    model_dumpdir
    #                    +'crystal_dataset'
    #                    +f'_epochs{epochs}_batchsize{batch_size}_dropout{dropout_rate}_' 
    #                    + layer_names)
    #            model_evaluation_dumpname = (
    #                    model_evaluation_dumpdir
    #                    +'crystal_dataset'
    #                    +f'_epochs{epochs}_batchsize{batch_size}_dropout{dropout_rate}_' 
    #                    + layer_names)
    #            history_dumpname = (
    #                    model_evaluation_dumpdir
    #                    +'crystal_dataset'
    #                    +f'_epochs{epochs}_batchsize{batch_size}_dropout{dropout_rate}_'
    #                    +layer_names
    #                    +'_history')

    #            if model_exists(model_dumpname) and not recompute_model:
    #                print('======Model exists======')
    #                with open(model_dumpname+'.json', 'r') as json_file:
    #                    loaded_json_file = json_file.read()
    #                    model = model_from_json(loaded_json_file)
    #                    model.load_weights(model_dumpname+'.h5')
    #                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #            else:
    #                print('=======Calculating Model========')
    #                model = run_model(
    #                        X_train, X_test, y_train, y_test, model_layers,
    #                        model_dumpname, epochs, history_dumpname,
    #                        dropout_rate=dropout_rate, batch_size=batch_size)

    #            evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    #            print('EVALUATION: ', evaluation)
    #            if evaluation[0] < best_evaluation[0]:
    #                best_evaluation = evaluation
    #                best_model = model
    #                print('NEW BEST EVALUATION: ', evaluation)
    #            np.save(model_evaluation_dumpname, model_layers)
    #            if run_predict:
    #                for datafile, adjacencyfile, dumpname in zip(data_files, adjacency_files, dump_names):
    #                    data = np.load(adjacencyfile)
    #                    data = data.reshape(-1, np.prod(data.shape[1:]))
    #                    print(data.shape)
    #                    StandardScaler(copy=False).fit_transform(data)
    #                    prediction = test_model(model, data, tol=0.01)
    #                    np.save('prediction_testing', prediction)
    #                    prediction, labels = map_to_labels(prediction, crystal_labels, keep_biggest=None)
    #                    np.save(dumpname+'_clusters', Counter(prediction))
    #                    np.save(dumpname+'_labels', labels)
    #                    max_prediction = Counter(prediction)
    #                    write_data(datafile, prediction, dumpname)
