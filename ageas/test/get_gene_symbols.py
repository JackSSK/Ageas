# import mygene
# import pandas as pd
# import ageas.tool.json as json
#
#
# def _update_grn(grn, symbol_dict):
# 	for id, record in grn['genes'].items():
# 		symbol = symbol_dict[id]
# 		if record['symbol'] != symbol and record['symbol'] == id:
# 			record['symbol'] = symbol
# 		else:
# 			print('WTF?')
#
# def get_gene_symbols(path):
# 	# Get Meta GRN, all Gens should be included here
# 	if path[-1] != '/': path += '/'
# 	key_atlas = json.decode(path + 'key_atlas.js')
# 	full_atlas = json.decode(path + 'full_atlas.js')
# 	psGRNs = json.decode(path + 'pseudo_sample_GRNs.js')
# 	report = pd.read_csv(path + 'report.csv', header = 0)
# 	meta_grn = json.decode(path + 'meta_GRN.js')
# 	meta_report = pd.read_csv(path + 'meta_report.csv', header = 0)
#
# 	mygene_query = mygene.MyGeneInfo().querymany(
# 		meta_grn['genes'].keys(),
# 		scopes = 'ensembl.gene'
# 	)
# 	symbol_dict = dict()
# 	for ele in mygene_query:
# 		# only take one gene symbol from mygene
# 		if ele['query'] not in symbol_dict:
# 			symbol_dict[ele['query']] = None
# 		else:
# 			continue
# 		# add gene symbol to dict
# 		try:
# 			symbol_dict[ele['query']] = ele['symbol']
# 		except:
# 			symbol_dict[ele['query']] = None
# 	del mygene_query
#
# 	# Change Gene Symbols for GRNs
# 	_update_grn(meta_grn, symbol_dict)
# 	_update_grn(key_atlas, symbol_dict)
# 	for id, network in full_atlas.items():
# 		_update_grn(network, symbol_dict)
# 	for id, grn in psGRNs['class1'].items():
# 		_update_grn(grn, symbol_dict)
# 	for id, grn in psGRNs['class2'].items():
# 		_update_grn(grn, symbol_dict)
#
# 	# Change Gene Symbols in Reports
# 	report['Gene Symbol'] = [symbol_dict[id] for id in report['ID']]
# 	meta_report['Gene Symbol'] = [symbol_dict[id] for id in meta_report['ID']]
#
# 	# Save files
# 	json.encode(meta_grn, path + 'meta_GRN.js')
# 	json.encode(key_atlas, path + 'key_atlas.js')
# 	json.encode(full_atlas, path + 'full_atlas.js')
# 	json.encode(psGRNs, path + 'pseudo_sample_GRNs.js')
# 	report.to_csv(path + 'report.csv', header = True, index = None)
# 	meta_report.to_csv(path + 'meta_report.csv', header = True, index = None)
#
# if __name__ == '__main__':
# 	path = 'p28cm_p28ncm/std==d/'
# 	path = 'p4_p28/std==d/'
# 	get_gene_symbols(path)
