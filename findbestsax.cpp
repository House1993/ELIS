//
// Created by house on 3/31/17.
//

#include <cstring>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <cmath>
#include <stdlib.h>
#include <unordered_set>

using namespace std;
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

void CreateSAXList(const vector<vector<double> > &Data, int subseq_len, int sax_len, int w) {
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

    for (obj_id = 0; obj_id < (int) Data.size(); obj_id++) {
        ex = ex2 = 0;
        prev_word = -1;

        for (k = 0; k < sax_len; k++)
            sum_segment[k] = 0;

        /// Case 1: Initial
        for (j = 0; (j < (int) Data[obj_id].size()) && (j < subseq_len); j++) {
            d = Data[obj_id][j];
            ex += d;
            ex2 += d * d;
            slot = (int) floor((j) / w);
            sum_segment[slot] += d;
        }

        /// Case 2: Slightly Update
        for (; (j <= (int) Data[obj_id].size()); j++) {
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
            if (j < (int) Data[obj_id].size()) {
                ex -= Data[obj_id][j_st];
                ex2 -= Data[obj_id][j_st] * Data[obj_id][j_st];

                for (k = 0; k < sax_len - 1; k++) {
                    sum_segment[k] -= Data[obj_id][j_st + (k) * w];
                    sum_segment[k] += Data[obj_id][j_st + (k + 1) * w];
                }
                sum_segment[k] -= Data[obj_id][j_st + (k) * w];
                sum_segment[k] += Data[obj_id][j_st + min((k + 1) * w, subseq_len)];

                d = Data[obj_id][j];
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
void SortAllSAX(const vector<int> &y, const int c, const int length, const int top_k) {
    USAX_Map_type::iterator it;
    USAX_elm_type usax;
    candidatesax can;
    can.length = length;

    vector<double> c_in(c, 0);
    int sum;

    for (it = USAX_Map.begin(); it != USAX_Map.end(); it++) {
        usax = it->second;

        int fid = rand() % usax.sax_id.size();
        can.sax_id.first = usax.sax_id[fid].first;
        can.sax_id.second = usax.sax_id[fid].second;

        sum = 0;
        for (Obj_count_type::iterator o_it = usax.obj_count.begin(); o_it != usax.obj_count.end(); o_it++) {
            int cid = y[o_it->first];
            int count = o_it->second;
            c_in[cid] += (count);
            sum += count;
        }

        for (int i = 0; i < c; ++i) {
            can.score = (double) (c_in[i]) / sum;
            c_in[i] = 0;
            if (que[i].size() < top_k || que[i].top().score < can.score) {
                que[i].pop();
                que[i].push(can);
            }
        }

    }
}

void findbestsax(const vector<vector<double> > &x, const vector<int> &y, const int len, const int c,
                 const int top_k, const int step, vector<vector<pair<int, pair<int, int> > > > &ans) {
    for (int i = step; i < len; i += step) {

        /**
         * parameters
         */
        int sax_len = 15; // sax_max_len;
        int R = 10;
        double percent_mask = 0.25;

        /// Make w and sax_len both integer
        int w = (int) ceil(1.0 * i / sax_len);
        sax_len = (int) ceil(1.0 * i / w);

        CreateSAXList(x, i, sax_len, w);

        RandomProjection(R, percent_mask, sax_len);

        SortAllSAX(y, c, i, top_k);
    }

    for (int i = 0; i < c; ++i) {
        ans.push_back(vector<pair<int, pair<int, int> > >());
        while (!que[i].empty()) {
            candidatesax can = que[i].top();
            que[i].pop();
            ans[i].push_back(make_pair(can.length, can.sax_id));
        }
    }
}