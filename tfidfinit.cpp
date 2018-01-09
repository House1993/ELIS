//
// Created by house on 5/23/17.
//

#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <set>
#include <map>

using namespace std;

#define D_NUM 5
#define DEFAULT_PAA_DIMENSION 10
#define PAA_BUCKET 10

#define INF 200000
#define MAX_CHAR_PER_LINE 200000
#define MAX_CLASS 100

int PAA_DIMENSION[D_NUM] = {2, 5, 10, 20, 50};
int FUZZY = 1;

// train/test data
vector<vector<double> > data_x; // data_n * ts_len
double max_data = -INF, min_data = INF, bucket_w;
vector<int> label;
int datacnt[MAX_CLASS];
int shapelets_n[MAX_CLASS];

int ts_len; // time series length
int data_n; // number of data
int num_c; // number of class

struct point {
    vector<int> f;
    int d;

    bool operator<(const point ff) const {
        for (int i = 0; i < d; ++i) {
            if (f[i] != ff.f[i]) {
                return f[i] < ff.f[i];
            }
        }
        return false;
    }
};

map<point, set<int> > tree[D_NUM][MAX_CLASS];

struct paaword {
    double score;
    int window;
    vector<int> vec;
    set<int> cov;
    set<int> allcov;

    bool operator<(const paaword f) const {
        if (fabs(score - f.score) > 1e-8) {
            return score > f.score;
        }
        if (cov.size() != f.cov.size()) {
            return cov.size() > f.cov.size();
        }
        return window * vec.size() < f.window * f.vec.size();
    }

    bool operator==(const paaword f) const {
        if (window * vec.size() != f.window * f.vec.size()) {
            return false;
        }
        for (int i = 0; i < window * vec.size(); ++i) {
            if (abs(vec[i / window] - f.vec[i / f.window]) > 1) {
                return false;
            }
        }
        return true;
    }

};

vector<paaword> candidates[MAX_CLASS];
vector<paaword> results[MAX_CLASS];

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

    set<int> pool;
    vector<int> all_label;

    for (int i = 0; i < data_n; i++) {
        fgets(buff, MAX_CHAR_PER_LINE, f);
        tmp = strtok(buff, ", \r\n");

        double now_label = atoi(tmp);
        label.push_back(now_label);
        pool.insert(now_label);

        tmp = strtok(NULL, ", \r\n");
        for (int j = 0; j < ts_len; ++j) {

            double now_data = atof(tmp);
            max_data = max(max_data, now_data);
            min_data = min(min_data, now_data);

            data_x[i].push_back(now_data);
            tmp = strtok(NULL, ", \r\n");
        }
    }

    num_c = pool.size();
//    for (int item : pool) {
//        all_label.push_back(item);
//    }
    for (set<int>::iterator it = pool.begin(); it != pool.end(); ++it) {
        all_label.push_back(*it);
    }

    for (int i = 0; i < data_n; ++i) {
        label[i] = lower_bound(all_label.begin(), all_label.end(), label[i]) - all_label.begin();
        ++datacnt[label[i]];
    }

    bucket_w = (max_data - min_data) / PAA_BUCKET;
    max_data += bucket_w / 100;
    bucket_w = (max_data - min_data) / PAA_BUCKET;

    fclose(f);
}

double paa2ts(int x) {
    return min_data + bucket_w * (0.5 + x);
}

//void insertintohashmap(vector<int> &qu, int offset, int d, int tsid) {
void insertintohashmap(vector<int> &qu, int offset, int d, int tsid, int len, int start) {
    // PAA window
    int window = len / PAA_DIMENSION[d];
    double distance = 0;
    int s_pos = start + offset * window;
    for (int i = 0; i < PAA_DIMENSION[d]; ++i) {
        for (int j = 0; j < window; ++j) {
            double tmp_dis = paa2ts(qu[offset + i]) - data_x[tsid][s_pos + i * window + j];
            distance += tmp_dis * tmp_dis;
        }
    }
    distance /= len;
    if (distance > bucket_w * bucket_w * FUZZY) {
        return;
    }
    point f;
    f.d = PAA_DIMENSION[d];
    for (int i = 0; i < PAA_DIMENSION[d]; ++i) {
        f.f.push_back(qu[offset + i]);
    }
    tree[d][label[tsid]][f].insert(tsid);
}

set<int> queryfromhashmap(vector<int> &qu, int offset, int d, int treeid) {
    set<int> cnt;
    point f;
    f.d = PAA_DIMENSION[d];
    for (int i = 0; i < PAA_DIMENSION[d]; ++i) {
        f.f.push_back(qu[offset + i]);
    }
    if (tree[d][treeid].count(f)) {
        cnt.insert(tree[d][treeid][f].begin(), tree[d][treeid][f].end());
    }
    for (int i = 0; i < PAA_DIMENSION[d]; ++i) {
        f.f[i] -= 1;
        if (tree[d][treeid].count(f)) {
            cnt.insert(tree[d][treeid][f].begin(), tree[d][treeid][f].end());
        }
        f.f[i] += 2;
        if (tree[d][treeid].count(f)) {
            cnt.insert(tree[d][treeid][f].begin(), tree[d][treeid][f].end());
        }
        f.f[i] -= 1;
    }
    return cnt;
}

void createwords() {
    int shapelet_minlen = max(DEFAULT_PAA_DIMENSION, (ts_len / 20 + DEFAULT_PAA_DIMENSION - 1)
                                                     / DEFAULT_PAA_DIMENSION * DEFAULT_PAA_DIMENSION);
    for (int len = shapelet_minlen; len <= ts_len; len += shapelet_minlen) {
//        printf("search length %d:\n", len);

        // clear hash map
        for (int i = 0; i < D_NUM; ++i) {
            for (int j = 0; j <= num_c; ++j) {
                tree[i][j].clear();
            }
        }

        for (int d = 0; d < D_NUM; ++d) {
            if (len % PAA_DIMENSION[d] != 0) {
                continue;
            }

            // PAA window
            int window = len / PAA_DIMENSION[d];

            // insert words into hashmap
            for (int i = 0; i < data_n; ++i) {
                for (int start = 0; start < window; ++start) {
                    vector<int> qu;

                    // PAA
                    for (int offset = start; offset + window <= ts_len; offset += window) {
                        double avg = 0;
                        for (int j = offset; j < offset + window; ++j) {
                            avg += data_x[i][j];
                        }
                        int bucket = (int) floor((avg / window - min_data) / bucket_w - 1e-8);
                        qu.push_back(bucket);
                    }

                    // insert into hash map
                    for (int j = 0; j + PAA_DIMENSION[d] <= qu.size(); ++j) {
//                        insertintohashmap(qu, j, d, i);
                        insertintohashmap(qu, j, d, i, len, start);
                    }

                }
            }
//            printf("dimension %d insert finished\n", PAA_DIMENSION[d]);

            // query from hashmap to calculate tfidf score
            for (int i = 0; i < data_n; ++i) {
                for (int start = 0; start < window; ++start) {
                    vector<int> qu;

                    // PAA
                    for (int offset = start; offset + window <= ts_len; offset += window) {
                        double avg = 0;
                        for (int j = offset; j < offset + window; ++j) {
                            avg += data_x[i][j];
                        }
                        int bucket = (int) floor((avg / window - min_data) / bucket_w);
                        qu.push_back(bucket);
                    }

                    // query from hash map
                    for (int j = 0; j + PAA_DIMENSION[d] <= qu.size(); ++j) {

                        int inclasses = 0;

                        vector<set<int> > select;
                        set<int> allselect;
                        for (int k = 0; k < num_c; ++k) {
                            select.push_back(queryfromhashmap(qu, j, d, k));
                            if (select[k].size() != 0) {
                                ++inclasses;
                                allselect.insert(select[k].begin(), select[k].end());
                            }
                        }

                        if (inclasses != 1) {
                            continue;
                        }
                        double idf = log2(2.0);

//                        double idf = log2((double) (num_c) / inclasses);

                        for (int k = 0; k < num_c; ++k) {

                            int inthisclass = (int) select[k].size();

                            if (inthisclass == 0) {
                                continue;
                            }

                            double tf = (double) (inthisclass) / datacnt[k];

                            paaword now;
                            now.window = window;
                            now.score = tf * idf;
                            now.cov = select[k];
                            now.allcov = allselect;
                            for (int cp = 0; cp < PAA_DIMENSION[d]; ++cp) {
                                now.vec.push_back(qu[j + cp]);
                            }
                            candidates[k].push_back(now);

                        }
                    }
                }
            }
//            printf("dimension %d query finished\n", PAA_DIMENSION[d]);
        }
    }
}

double calculatedis(paaword paa, int tsid) {
    double dis = -1;
    int len = paa.window * paa.vec.size();
    for (int s_pos = 0; s_pos + len <= ts_len; ++s_pos) {
        double distance = 0;
        for (int i = 0; i < paa.vec.size(); ++i) {
            for (int j = 0; j < paa.window; ++j) {
                double tmp_dis = paa2ts(paa.vec[i]) - data_x[tsid][s_pos + i * paa.window + j];
                distance += tmp_dis * tmp_dis;
            }
        }
        distance /= len;
        if (dis < 0 || dis > distance) {
            dis = distance;
        }
    }
    return dis;
}

int check(int classid, vector<paaword> tmp_result, int ENOUGH_COVER) {
    int *cnt_cover = new int[data_n];
    for (int i = 0; i < data_n; ++i) {
        cnt_cover[i] = 0;
    }
//    for (paaword paa : tmp_result) {
    for (int i = 0; i < tmp_result.size(); ++i) {
        paaword paa = tmp_result[i];
        double max_dis = -1;
//        for (int cov : paa.cov) {
        for (set<int>::iterator cov_it = paa.cov.begin(); cov_it != paa.cov.end(); ++cov_it) {
            int cov = *cov_it;
            max_dis = max(max_dis, calculatedis(paa, cov));
            ++cnt_cover[cov];
        }
        max_dis += 1e-8;
        for (int i = 0; i < data_n; ++i) {
            if (!paa.cov.count(i)) {
                if (calculatedis(paa, i) < max_dis) {
                    ++cnt_cover[i];
                }
            }
        }
    }
//    printf("wrong cover ts:");
    int cnt_wrong = 0;
    for (int i = 0; i < data_n; ++i) {
        if (cnt_cover[i] >= ENOUGH_COVER && label[i] != classid) {
            ++cnt_wrong;
//            printf("%3d", i);
        }
    }
//    printf("\n");
    delete[] cnt_cover;
    return cnt_wrong;
}

void getresults() {
    for (int i = 0; i < num_c; ++i) {
//        printf("class %d:\n", i);

        sort(candidates[i].begin(), candidates[i].end());

        int best_wrong = data_n + 1;
        int best_cover = 1;
        results[i].clear();

        for (int ENOUGH_COVER = 1; ENOUGH_COVER <= 5; ++ENOUGH_COVER) {

            vector<paaword> tmp_result;
            tmp_result.clear();
            map<int, int> cover;
            cover.clear();
            int total_cover = 0;

//            for (paaword now : candidates[i]) {
            for (int now_it = 0; now_it < candidates[i].size(); ++now_it) {
                paaword now = candidates[i][now_it];

                bool repeat = true;
//                for (int cov : now.cov) {
                for (set<int>::iterator cov_it = now.cov.begin(); cov_it != now.cov.end(); ++cov_it) {
                    int cov = *cov_it;
                    if (cover[cov] < ENOUGH_COVER) {
                        repeat = false;
                        break;
                    }
                }
                if (repeat) {
                    continue;
                }

                bool chosen = true;
                for (int j = 0; j < tmp_result.size(); ++j) {
                    if (now == tmp_result[j]) {
//                        for (int cov : now.cov) {
                        for (set<int>::iterator cov_it = now.cov.begin(); cov_it != now.cov.end(); ++cov_it) {
                            int cov = *cov_it;
                            if (!tmp_result[j].cov.count(cov)) {
                                tmp_result[j].cov.insert(cov);
                                if (cover[cov] < ENOUGH_COVER) {
                                    ++cover[cov];
                                    ++total_cover;
                                }
                            }
                        }
                        chosen = false;
                        break;
                    }
                }
                if (chosen) {
                    tmp_result.push_back(now);
//                    for (int cov : now.cov) {
                    for (set<int>::iterator cov_it = now.cov.begin(); cov_it != now.cov.end(); ++cov_it) {
                        int cov = *cov_it;
                        if (cover[cov] < ENOUGH_COVER) {
                            ++cover[cov];
                            ++total_cover;
                        }
                    }
                }

                if (total_cover == datacnt[i] * ENOUGH_COVER) {
                    break;
                }
            }

//            printf("cover %d ", ENOUGH_COVER);
            int tmp_wrong = check(i, tmp_result, ENOUGH_COVER);
            if (tmp_wrong < best_wrong) {
                best_cover = ENOUGH_COVER;
                best_wrong = tmp_wrong;
                results[i] = tmp_result;
                shapelets_n[i] = tmp_result.size();
                if (best_wrong == 0) {
                    break;
                }
            }
        }
//        printf("cover %d is best!\n", best_cover);
    }
}

void findbestshape() {
    createwords();
    getresults();
}

void printshape(char *out_file) {
    FILE *f;
    f = fopen(out_file, "w");

    for (int i = 0; i < num_c; ++i) {
        fprintf(f, "%d\n", shapelets_n[i]);
//        for (paaword now : results[i]) {
        for (int j = 0; j < results[i].size(); ++j) {
            paaword now = results[i][j];
            fprintf(f, "%f,%d", -now.score, now.window * now.vec.size());
            for (int k = 0; k < now.window * now.vec.size(); ++k) {
                fprintf(f, ",%f", paa2ts(now.vec[k / now.window]));
            }
            fprintf(f, "\n");
//            fprintf(f, "vector\t\t");
//            for (int k = 0; k < now.vec.size(); ++k) {
//                fprintf(f, " %d", now.vec[k]);
//            }
//            fprintf(f, "\ncover\t\t");
//            for (set<int>::iterator cov = now.cov.begin(); cov != now.cov.end(); ++cov) {
//                fprintf(f, " %d", *cov);
//            }
//            fprintf(f, "\nall cover\t");
//            for (set<int>::iterator cov = now.allcov.begin(); cov != now.allcov.end(); ++cov) {
//                fprintf(f, " %d", *cov);
//            }
//            fprintf(f, "\n");
        }
    }

    fclose(f);
}

int main(int argc, char *argv[]) {

    /**
     * arguments:
     * ts_len data_file data_n [PAA fuzzy=1] [cover=1]
     */

    ts_len = atoi(argv[1]);
    char *data_file = argv[2];
    data_n = atoi(argv[3]);
    if (argc >= 5) {
        FUZZY = atoi(argv[4]);
        FUZZY *= FUZZY;
    }

    readfile(data_file);

    printf("reading data finished\n");

    double start_time = clock();
    findbestshape();
    double end_time = clock();

    printf("shapelet discovery finished\n");

    printshape("init.txt");

    printf("Total Running Time = %.3f sec\n", (end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}