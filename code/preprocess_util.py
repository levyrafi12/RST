def is_last_edu_in_sent(tree, edu_ind):
	if edu_ind == len(tree._EDUS_table) - 1:
		return True

	edu = tree._EDUS_table[edu_ind]
	if edu[-1] == '.' or edu[-2:] == '."' or edu[-1] == "?" or edu[-1] == "!" \
		or edu[-2:] == '.)' or edu[-3:] == '.")' or edu[-1] == ':':
		edu_next = tree._EDUS_table[edu_ind + 1]
		if not edu_next[0].islower():
			return True
	return False