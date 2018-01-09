#include <cstdio>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cmath>

using namespace std;

#define DEBUG_SAX false

#define MAX_CHAR_PER_LINE 200000

// train/test data
vector<vector<double> > data_x; // data_n * ts_len
vector<int> label;
vector<vector<double> > data_y;

int ts_len; // time series length
int data_n; // number of data
int num_c, min_class_id = MAX_CHAR_PER_LINE; // number of class , min class id (init inf)
int shapelet_n, shapelet_minlen; // number of shapelet , shapelet min length

#define MAX_CLASS 100

typedef unordered_set<int> Obj_set_type;
typedef vector<pair<int, int> > SAX_id_type;
typedef unordered_map<int, int> Obj_count_type;
struct USAX_elm_type {
    Obj_set_type obj_set;
    SAX_id_type sax_id;
    Obj_count_type obj_count;
};
typedef unordered_map<int, USAX_elm_type> USAX_Map_type;
typedef unordered_map<int, Obj_set_type> Hash_Mark_type;

struct candidatesax {
    double score;
    pair<int, int> sax_id;
    int length;

    bool operator<(const candidatesax f) const {
        return score > f.score;
    }

};

priority_queue<candidatesax> que[MAX_CLASS];
USAX_Map_type USAX_Map;

/// Fix card = 4 here !!!
int CreateSAXWord(const double *sum_segment, const int *elm_segment, double mean, double std, int sax_len) {
    int word = 0, val = 0;
    double d = 0;

    for (int i = 0; i < sax_len; i++) {
        d = (sum_segment[i] / elm_segment[i] - mean) / std;
        if (d < 0)
            if (d < -0.67) val = 0;
            else val = 1;
        else if (d < 0.67) val = 2;
        else val = 3;

        word = (word << 2) | (val);
    }
    return word;
}

void CreateSAXList(int subseq_len, int sax_len, int w) {
    double ex, ex2, mean, std;
    double sum_segment[sax_len];
    int elm_segment[sax_len];
    int obj_id, j, j_st, k, slot;
    double d;
    int word, prev_word;
    USAX_elm_type *ptr;

    for (k = 0; k < sax_len; k++)
        elm_segment[k] = w;
    elm_segment[sax_len - 1] = subseq_len - (sax_len - 1) * w;

    for (obj_id = 0; obj_id < (int) data_x.size(); obj_id++) {
        ex = ex2 = 0;
        prev_word = -1;

        for (k = 0; k < sax_len; k++)
            sum_segment[k] = 0;

        /// Case 1: Initial
        for (j = 0; (j < (int) data_x[obj_id].size()) && (j < subseq_len); j++) {
            d = data_x[obj_id][j];
            ex += d;
            ex2 += d * d;
            slot = (int) floor((j) / w);
            sum_segment[slot] += d;
        }

        /// Case 2: Slightly Update
        for (; (j <= (int) data_x[obj_id].size()); j++) {
            j_st = j - subseq_len;

            mean = ex / subseq_len;
            std = sqrt(ex2 / subseq_len - mean * mean);

            /// Create SAX from sum_segment
            word = CreateSAXWord(sum_segment, elm_segment, mean, std, sax_len);

            if (word != prev_word) {
                prev_word = word;
                ptr = &USAX_Map[word];
                ptr->obj_set.insert(obj_id);
                ptr->sax_id.push_back(std::move(make_pair(obj_id, j_st)));
            }

            /// For next update
            if (j < (int) data_x[obj_id].size()) {
                ex -= data_x[obj_id][j_st];
                ex2 -= data_x[obj_id][j_st] * data_x[obj_id][j_st];

                for (k = 0; k < sax_len - 1; k++) {
                    sum_segment[k] -= data_x[obj_id][j_st + (k) * w];
                    sum_segment[k] += data_x[obj_id][j_st + (k + 1) * w];
                }
                sum_segment[k] -= data_x[obj_id][j_st + (k) * w];
                sum_segment[k] += data_x[obj_id][j_st + min((k + 1) * w, subseq_len)];

                d = data_x[obj_id][j];
                ex += d;
                ex2 += d * d;
            }
        }
    }
}

/// create mask word (two random may give same position, we ignore it)
int CreateMaskWord(int num_mask, int word_len) {
    int a, b;
    a = 0;
    for (int i = 0; i < num_mask; i++) {
        b = 1 << (rand() % word_len);
        a = a | b;
    }
    return a;
}

/// Count the number of occurrences
void RandomProjection(int R, double percent_mask, int sax_len) {
    Hash_Mark_type Hash_Mark;

    USAX_Map_type::iterator it;
    int word, mask_word, new_word;
    Obj_set_type *obj_set, *ptr;
    Obj_set_type::iterator o_it;

    int num_mask = ceil(percent_mask * sax_len);

    for (int r = 0; r < R; r++) {
        mask_word = CreateMaskWord(num_mask, sax_len);

        /// random projection and mark non-duplicate object
        for (it = USAX_Map.begin(); it != USAX_Map.end(); it++) {
            word = it->first;
            obj_set = &(it->second.obj_set);

            new_word = word | mask_word;
            ptr = &Hash_Mark[new_word];
            ptr->insert(obj_set->begin(), obj_set->end());
        }

        /// hash again for keep the count
        for (it = USAX_Map.begin(); it != USAX_Map.end(); it++) {
            word = it->first;
            new_word = word | mask_word;
            obj_set = &(Hash_Mark[new_word]);
            for (o_it = obj_set->begin(); o_it != obj_set->end(); o_it++) {
                (it->second.obj_count[*o_it])++;
            }
        }
        Hash_Mark.clear();
    }
}

/// Sort each SAX
void SortAllSAX(const int length, const int R) {
    USAX_Map_type::iterator it;
    USAX_elm_type usax;
    candidatesax can;
    can.length = length;

    vector<double> c_in(num_c, 0);
    vector<double> c_out(num_c, 0);
//    int sum;
    int sumin, sumout;

    for (it = USAX_Map.begin(); it != USAX_Map.end(); it++) {
        usax = it->second;

        int fid = rand() % usax.sax_id.size();
        can.sax_id.first = usax.sax_id[fid].first;
        can.sax_id.second = usax.sax_id[fid].second;

//        sum = 0;
        sumin = sumout = 0;
        for (Obj_count_type::iterator o_it = usax.obj_count.begin(); o_it != usax.obj_count.end(); o_it++) {
            int cid = label[o_it->first];
            int count = o_it->second;
            c_in[cid] += (count);
            c_out[cid] += (R - count);
//            sum += count;
            sumin += count;
            sumout += R - count;
        }

        if (DEBUG_SAX) {
            printf("%d,%d,%d,%d", it->first, can.length, sumin, sumout);
        }

        for (int i = 0; i < num_c; ++i) {
//            can.score = (double) (c_in[i]) / sum;
//            can.score = abs((c_in[i] + sumout - c_out[i]) - (c_out[i] + sumin - c_in[i]));
            can.score = (c_in[i] + sumout - c_out[i]) - (c_out[i] + sumin - c_in[i]);

            if (DEBUG_SAX) {
                printf(",%f,%f,%f,%f", c_in[i], c_out[i], c_in[i] / sumin, can.score);
            }

            c_in[i] = c_out[i] = 0;
            if (que[i].size() < shapelet_n || que[i].top().score < can.score) {
                que[i].push(can);
                if (que[i].size() > shapelet_n) {
                    que[i].pop();
                }
            }
        }

        if (DEBUG_SAX) {
            printf("\n");
        }

    }
}

void findbestsax() {
    for (int i = shapelet_minlen; i < ts_len; i += shapelet_minlen) {

        /**
         * parameters
         */
        int sax_len = 15; // sax_max_len;
        int R = 10;
        double percent_mask = 0.25;

        /// Make w and sax_len both integer
        int w = (int) ceil(1.0 * i / sax_len);
        sax_len = (int) ceil(1.0 * i / w);

        CreateSAXList(i, sax_len, w);

        RandomProjection(R, percent_mask, sax_len);

        SortAllSAX(i, R);

        USAX_Map.clear();

    }
}

/**
 * read file data( n lines, len + 1 numbers each line )
 * first number is label, next len numbers are x
 */
void readfile(char *data_file) {
    FILE *f;
    f = fopen(data_file, "r");

    char buff[MAX_CHAR_PER_LINE];
    char *tmp;

    for (int i = 0; i < data_n; ++i) {
        data_x.push_back(vector<double>());
    }

    num_c = 0;

    for (int i = 0; i < data_n; i++) {
        fgets(buff, MAX_CHAR_PER_LINE, f);
        tmp = strtok(buff, ", \r\n");

        label.push_back(atoi(tmp));

        min_class_id = min(min_class_id, label[i]);

        tmp = strtok(NULL, ", \r\n");
        for (int j = 0; j < ts_len; ++j) {
            data_x[i].push_back(atof(tmp));
            tmp = strtok(NULL, ", \r\n");
        }
    }

    for (int i = 0; i < data_n; ++i) {
        label[i] -= min_class_id;
        num_c = max(num_c, label[i]);
    }

    ++num_c;

    fclose(f);
}

void printsax(char *out_file) {
    FILE *f;
    f = fopen(out_file, "w");

    for (int i = 0; i < num_c; ++i) {
        double score_max = que[i].top().score;
        int j = 0;
        for (; j < shapelet_n && !que[i].empty(); ++j) {
            candidatesax now = que[i].top();
            que[i].pop();
            fprintf(f, "%f,%d", now.score / score_max, now.length);

            double ex = 0, ex2 = 0;
            for (int k = now.sax_id.second; k < now.sax_id.second + now.length; ++k) {
                double d = data_x[now.sax_id.first][k];
                ex += d;
                ex2 += d * d;
            }
            double mean = ex / now.length;
            double std = sqrt(ex2 / now.length - mean * mean);

            for (int k = now.sax_id.second; k < now.sax_id.second + now.length; ++k) {
                fprintf(f, ",%f", (data_x[now.sax_id.first][k] - mean) / std);
            }
            fprintf(f, "\n");
        }
        if (j < shapelet_n) {
            printf("class %d has need %d shapelet more\n", i, shapelet_n - j);
        }
    }

    fclose(f);
}

int main(int argc, char *argv[]) {

    /**
     * arguments:
     * 1.train ts_len data_file data_n
     * 2.test ts_len data_file data_n
     */
    char *operation = argv[1];
    ts_len = atoi(argv[2]);

    /**
     * init shapelet length and step
     */
    shapelet_minlen = max(15, ts_len / 20);

    char *data_file = argv[3];
    data_n = atoi(argv[4]);

    shapelet_n = atoi(argv[5]);

    readfile(data_file);

    double start_time = clock();
    findbestsax();
    double end_time = clock();

    printsax("init.txt");

    printf("Total Running Time = %.3f sec\n", (end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}