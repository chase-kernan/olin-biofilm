
import tables

def merge(base_file, files_to_merge):
    #with tables.open_file(files_to_merge[0], 'r') as first:
    #    first.copy_file(base_file, overwrite=True)

    with tables.open_file(base_file, 'w') as base:
        for path in files_to_merge:
            print path
            with tables.open_file(path, 'r') as source:
                for table in source.walk_nodes(classname='Table'):
                    if table._v_pathname in base:
                        base_table = base.get_node(table._v_pathname)
                        merge_table_into(table, base_table)
                    else:
                        copy_table_into(table, base)

def copy_table_into(table, base):
    print 'copying table', table
    parent = table._v_parent
    if parent._v_pathname != '/' and parent._v_pathname not in base:
        base.create_group(parent.parentnode._v_pathname, 
                          parent._v_name, 
                          createparents=True)
    table._f_copy(base.get_node(parent._v_pathname),
                  filters=tables.Filters(complib='blosc', complevel=1),
                  chunkshape=(1,))

def merge_table_into(table, base_table):
    print 'merging', table, 'into', base_table
    base_table.append(table.read())

if __name__ == '__main__':
    #import sys
    #merge(sys.argv[1], sys.argv[2:])

    merge('merged_sample_new.h5', ['sample_new{0}.h5'.format(i) for i in range(32)])