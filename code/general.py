import glob
import os

SEP = os.sep

def build_infile_name(fname, base_path, dis_files_dir, suffs):
	for suf in suffs:
		fn = build_file_name(fname, base_path, dis_files_dir, suf)
		if os.path.exists(fn):
			return fn
	assert False, "File input does not exist: " +  \
		SEP.join([base_path, dis_files_dir, fname]) + \
		" with possible suffices " + "|".join(suffs)
	return None

def build_file_name(base_fn, base_path, files_dir, suf):
	fn = base_path
	fn += SEP
	fn += files_dir
	fn += SEP
	fn += base_fn
	if suf != '':
		fn += "."
		fn += suf
	return fn

def create_dir(base_path, outdir):
	remove_dir(base_path, outdir)
	path = base_path
	path += SEP
	path += outdir
	os.makedirs(path)
	return path

def remove_dir(base_path, dir):
	path = base_path
	path += SEP
	path += dir
	if os.path.isdir(dir):
		path_to_files = path
		path_to_files += SEP + "*"
		for fn in glob.glob(path_to_files):
			os.remove(fn)
		os.rmdir(path)