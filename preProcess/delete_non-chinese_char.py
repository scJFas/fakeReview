from preProcess import import_meta_data as imd

OUTPUT_FILE = "keep_only_chinese.csv"

def delete_no_chinese(meta_data, key = 'reviewbody'):
    for i in range(len(meta_data[key]) ):
        s = ''
        for j in range(len(meta_data[key][i]) ):
            if not '\u4e00' <= meta_data[key][i][j] <= '\u9fa5':
                s += '/'
            else:
                s += meta_data[key][i][j]
        meta_data[key][i] = s

def main():
    meta_data = imd.import_meta_data();
    delete_no_chinese(meta_data)

    meta_data.to_csv(OUTPUT_FILE, header=True, index=False)

if __name__ == "__main__":
    main()
