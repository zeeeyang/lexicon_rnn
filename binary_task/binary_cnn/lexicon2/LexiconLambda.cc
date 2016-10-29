#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <unordered_map>
#include <unordered_set>
//#include <math.h>

using namespace std;
using namespace cnn;

//float pdrop = 0.02;
float pdrop = 0.5;
float punk = 0.1;
bool eval = false;
bool verbose = false;
unsigned INPUT_DIM = 64;
unsigned HIDDEN_DIM = 150;
unsigned TAG_HIDDEN_DIM = 64;
unsigned LAYERS = 1;
unsigned VOCAB_SIZE = 0;
unsigned LABEL_SIZE = 2;

cnn::Dict word_dict;

int kUNK; //tzy
unordered_map<string, vector<float> > pretrained_embeddings;
vector<float> avg_embeddings;

unordered_map<string, float> lex_dict;
unordered_set<int> singletons;

class Example {
public:
    vector<string> words;
    vector<string> lowercased_words;
    vector<int> word_ids; //after unk
    vector<int> real_word_ids;//before unk
    string label_str;
    int label;
};


struct LSTMClassifier {
    LookupParameters* p_word;

    Parameters* p_l2rR;
    Parameters* p_r2lR;
    Parameters* p_bias;

    Parameters* p_lbias;
    Parameters* p_tag2label;

    Parameters* p_start;
    Parameters* p_end;

    LSTMBuilder l2rbuilder;
    LSTMBuilder r2lbuilder;

    Parameters* p_lambda_shift;
    Parameters* p_lambda_lstm;
    Parameters* p_lambda_bias;

    Parameters* p_score_l2rR;
    Parameters* p_score_r2lR;
    Parameters* p_score_bias;
    Parameters* p_score_lbias;
    Parameters* p_score_tag2label;
    float zero = 0;
    float one = 1.0;

    explicit LSTMClassifier(Model& model) :
        l2rbuilder(LAYERS, INPUT_DIM , HIDDEN_DIM, &model),
        r2lbuilder(LAYERS, INPUT_DIM , HIDDEN_DIM, &model)
    {
        p_word   = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});

        p_l2rR = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
        p_r2lR = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
        p_bias = model.add_parameters({TAG_HIDDEN_DIM});

        p_lambda_shift = model.add_parameters({1, 2*HIDDEN_DIM});
        p_lambda_lstm = model.add_parameters({1, 2*HIDDEN_DIM});
        p_lambda_bias = model.add_parameters({1});


        p_lbias = model.add_parameters({1});
        p_tag2label = model.add_parameters({1, TAG_HIDDEN_DIM});

        p_start = model.add_parameters({INPUT_DIM});
        p_end = model.add_parameters({INPUT_DIM});
        p_score_l2rR = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
        p_score_r2lR = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
        p_score_bias = model.add_parameters({TAG_HIDDEN_DIM});
        p_score_lbias = model.add_parameters({1});
        p_score_tag2label = model.add_parameters({1, TAG_HIDDEN_DIM});



        for( size_t  word_index = 0; word_index <  VOCAB_SIZE; word_index++)
        {
            auto word_iter = pretrained_embeddings.find( word_dict.Convert(word_index) );
            if(word_iter != pretrained_embeddings.end())
            {
                ((LookupParameters*)p_word)->Initialize(word_index, word_iter->second);
            }
        }
        ((LookupParameters*)p_word)->Initialize(kUNK, avg_embeddings);

    }

    // return Expression of total loss
    Expression BuildGraph(Example& example, ComputationGraph& cg) {
        const vector<int>& sent = example.word_ids;
        const vector<string>& sent_words = example.words;
        const vector<string>& lowercased_words = example.lowercased_words;
        const unsigned slen = sent.size() ;
        l2rbuilder.new_graph(cg);  // reset builder for new graph
        l2rbuilder.start_new_sequence();

        r2lbuilder.new_graph(cg);  // reset builder for new graph
        r2lbuilder.start_new_sequence();

        Expression i_l2rR = parameter(cg, p_l2rR); // hidden -> word rep parameter
        Expression i_r2lR = parameter(cg, p_r2lR);
        Expression i_bias = parameter(cg, p_bias);  // word bias
        Expression i_lbias =  parameter(cg, p_lbias);
        Expression tag2label = parameter(cg, p_tag2label);

        Expression i_score_l2rR = parameter(cg, p_score_l2rR); // hidden -> word rep parameter
        Expression i_score_r2lR = parameter(cg, p_score_r2lR);
        Expression i_score_bias = parameter(cg, p_score_bias);  // word bias
        Expression i_score_lbias =  parameter(cg, p_score_lbias);
        Expression score_tag2label = parameter(cg, p_score_tag2label);

        Expression lambda_shift = parameter(cg, p_lambda_shift);
        Expression lambda_lstm = parameter(cg, p_lambda_lstm);
        Expression lambda_bias = parameter(cg, p_lambda_bias);


        Expression word_start = parameter(cg, p_start);
        Expression word_end = parameter(cg, p_end);


        vector<Expression> input_exprs;

        l2rbuilder.add_input(word_start);
        for (unsigned t = 0; t < slen; ++t) {
            assert(sent[t]!=-1);
            Expression i_word_t = lookup(cg, p_word, sent[t]);
            if (!eval) {
                i_word_t = dropout(i_word_t, pdrop);
            }
            input_exprs.push_back(i_word_t);
            Expression i_y_t = l2rbuilder.add_input(i_word_t);
        }
        l2rbuilder.add_input(word_end);

        r2lbuilder.add_input(word_end);
        for (unsigned t = 0; t < slen; ++t) {
            r2lbuilder.add_input(input_exprs[slen-1-t]);
        }
        r2lbuilder.add_input(word_start);

        vector<Expression> alpha_i(slen), word_scores(slen);
        vector<Expression> shift_scores;
        if(eval&& verbose)
        {
            copy(sent_words.begin(), sent_words.end(), ostream_iterator<string>(cerr," "));
            cerr<<endl;
        }
        vector<Expression> shift_reps;
        for (int t = 1; t <= slen; ++t) {
            Expression h_l2r = l2rbuilder.get_h((RNNPointer)t).back();
            Expression h_r2l = r2lbuilder.get_h((RNNPointer)t).back();
            bool lower_found = false;
            bool found = (lex_dict.find(sent_words[t-1]) != lex_dict.end());
            if(!found)
            {
                found = (lex_dict.find(lowercased_words[t-1]) != lex_dict.end());
                lower_found = found;
            }
            if(!found)
                word_scores[t-1] = input(cg, zero);
            else {
                word_scores[t-1] = input(cg, lower_found ? &lex_dict[lowercased_words[t-1]] : &lex_dict[sent_words[t-1]]);
            }
            Expression i_score_r_t =  tanh( i_score_bias + i_score_l2rR * h_l2r + i_score_r2lR * h_r2l );
            alpha_i[t-1] = 2.0 * tanh( i_score_lbias + score_tag2label * i_score_r_t );
            if(found)
            {
                shift_scores.push_back( alpha_i[t-1] * word_scores[t-1] );
                shift_reps.push_back(concatenate({h_l2r, h_r2l}));
                if(eval&&verbose)
                {
                    cg.incremental_forward();
                    cerr<< "[word] "<<t<<",\t"<< sent_words[t-1]<<":\t"<<(*alpha_i[t-1].value()) << " * " << (*word_scores[t-1].value()) <<"\t"<< (*shift_scores.back().value())<<endl;
                }
            }
        }
        Expression i_r_t =  tanh(i_bias + i_l2rR * l2rbuilder.back() + i_r2lR * r2lbuilder.back());
        Expression i_lstm_t = 2.0 *tanh(i_lbias + tag2label * i_r_t);//add tanh
        Expression total_score = i_lstm_t;
        if(eval&&verbose)
        {
            cerr<<"bias:\t"<< as_scalar(cg.incremental_forward())<<endl;
        }
        if(shift_reps.size() >0 ) {
            Expression shift_rep = average(shift_reps);
            Expression lstm_rep = concatenate({l2rbuilder.back(), r2lbuilder.back()});
            Expression lambda = logistic( lambda_shift * shift_rep + lambda_lstm * lstm_rep  + lambda_bias );
            if(eval&&verbose)
            {
                cerr<<"lambda:\t"<< as_scalar(cg.incremental_forward())<<endl;
            }
            Expression shift_score = average(shift_scores);
            if(eval&&verbose)
            {
                cerr<<"shift:\t"<< as_scalar(cg.incremental_forward())<<endl;
            }
            total_score = lambda * shift_score + (1.0 - lambda) * i_lstm_t;
        }
        if(eval&&verbose)
        {
            cerr<<"averaged score:\t"<< as_scalar(cg.incremental_forward())<<endl;
        }

        Expression positive_prob = logistic(total_score);
        if(eval&&verbose)
        {
            auto prob_value = as_scalar(cg.incremental_forward());
            cerr<<"positive prob: "<<prob_value<<endl;
            cerr<<"gold label: " << example.label<<endl;
            if(prob_value > 0.5 && example.label == 0)
                cerr<<"[ERROR]"<<endl;
            else if(prob_value < 0.5 && example.label == 1)
                cerr<<"[ERROR]"<<endl;
            else
                cerr<<"[RIGHT]" << endl;
            cerr<<endl<<endl;
        }
        Expression i_l_t = concatenate({1.0-positive_prob, positive_prob});
        return i_l_t;
    }
};


bool IsCurrentPredictionCorrection(ComputationGraph& cg, int y_true) {
    auto v = as_vector(cg.incremental_forward());
    assert(v.size() > 1);
    int besti = 0;
    float best = v[0];
    for (unsigned i = 1; i < v.size(); ++i)
        if (v[i] > best) {
            best = v[i];
            besti = i;
        }
    return (besti == y_true);
}

Expression CrossEntropyLoss(const Expression& y_pred, int y_true) {
    Expression lp = log(y_pred);
    Expression nll = -pick(lp, y_true);
    return nll;
}

void ReadEmbeddings(const string& input_file)
{
    ifstream fin(input_file);
    assert(fin);
    string line;
    bool first = true;
    int lines = 0;
    while(getline(fin, line))
    {
        lines++;
        istringstream sin(line);
        string word;
        sin>>word;
        float value;
        vector<float> vecs;
        while(sin>>value)
            vecs.push_back(value);
        if(first)
        {
            first = false;
            INPUT_DIM = vecs.size();
            avg_embeddings = vecs;
        }
        else {
            for(int i = 0; i< INPUT_DIM; i++)
                avg_embeddings[i] += vecs[i];
        }
        pretrained_embeddings[word] = vecs;
        word_dict.Convert(word);
    }
    fin.close();
    for(int i = 0; i< INPUT_DIM; i++)
    {
        avg_embeddings[i] /= lines;
    }
}

void ReadLexicons(const string& input_file)
{
    ifstream fin(input_file);
    assert(fin);
    string word;
    float score;
    while(fin>>word>>score)
    {
        if(score != 2)
            lex_dict[word] = score-2;
        //cerr<< word<<"\t"<<score << endl;
    }
    cerr<< "#Lex Num: "<< lex_dict.size() << endl;
    fin.close();
}

void ReadExample(const std::string& line, Example& example, bool debug = false)
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";

    bool is_word = true;
    while(in) {
        in >> word;
        if(debug)
            cerr<< word << endl;
        if (!in) break;
        if (word == sep) {
            is_word = false;
            if(debug) cerr<<"**CHANGE**"<<endl;
            continue;
        }
        if(is_word)
        {
            example.words.push_back(word);
        }
        else {
            istringstream sin(word);
            sin>>example.label;
            example.label -= 2; //shift label
            example.label_str = word;
        }
    }
}

void ReadExamples(const string& inputFile, vector<Example>& examples, int& lines, int& tokens)
{
    string line;
    cerr << "Reading data from " << inputFile << "...\n";
    {
        ifstream in(inputFile);
        assert(in);
        while(getline(in, line)) {
            Example example;
            ReadExample(line, example);
            if(example.label == 0)
                continue;
            if(example.label >0)
                example.label = 1;
            else
                example.label = 0;
            ++lines;
            example.word_ids.resize(example.words.size());
            example.lowercased_words.resize(example.words.size());
            for(size_t word_index =0; word_index < example.words.size(); word_index++)
            {
                string lower_rep = example.words[word_index];
                for(size_t i = 0; i< lower_rep.size(); i++)
                    if(lower_rep[i]<='Z' && lower_rep[i]>='A')
                        lower_rep[i] = lower_rep[i]-('Z'-'z');

                example.lowercased_words[word_index] = lower_rep;

                if( pretrained_embeddings.find(example.words[word_index]) == pretrained_embeddings.end() )
                {
                    bool lower_case_found2 = ( pretrained_embeddings.find(lower_rep) != pretrained_embeddings.end());
                    if( lower_case_found2 ) //always prefer pretrained embedding
                    {
                        cerr<<"[LOWER]" << example.words[word_index] << "\t" << lower_rep << endl;
                        example.word_ids[word_index] = word_dict.Convert(lower_rep);
                        //example.words[word_index] = lower_rep;
                    }
                    else {
                        example.word_ids[word_index] = word_dict.Convert(example.words[word_index]);
                    }
                }
                else {
                    example.word_ids[word_index] = word_dict.Convert(example.words[word_index]);
                }
            }
            example.real_word_ids = example.word_ids;
            examples.push_back(example);
            tokens += example.words.size();
        }
        cerr << lines << " lines, " << tokens << " tokens\n";
    }
}

void evaluate(vector<Example>& examples, LSTMClassifier& lstmClassifier, float& acc)
{
    eval = true;
    float num_correct = 0;
    float loss = 0;
    for (auto& sent : examples) {
        const int y = sent.label;
        ComputationGraph cg;
        Expression y_pred = lstmClassifier.BuildGraph(sent, cg);
        if (IsCurrentPredictionCorrection(cg, y)) num_correct++;
        CrossEntropyLoss(y_pred, y);
        loss += as_scalar(cg.incremental_forward());
    }
    acc = num_correct/ examples.size();
    cerr<<"Loss:"<< loss/ examples.size() << endl;
    cerr<<"Accuracy:"<< num_correct <<"/" << examples.size()<<" "<< acc << endl;
    eval = false;
}

int main(int argc, char** argv) {
    cerr<<"Command: " << endl;
    for(int i = 0; i< argc; i++)
        cerr<< i<<":\t" << argv[i] << endl; 
    if ( argc != 7) {
        cerr << "Usage: " << argv[0] << " vec lex train.phrase train dev test\n";
        return 1;
    }
    //cnn::Initialize(argc, argv);
    cnn::Initialize(argc, argv, 1887293186);
    //cnn::Initialize(argc, argv, 3657233929);
    vector<Example> training_examples, dev_examples, test_examples, training_sent_examples;
    string line;
    cerr << "Reading pretrained vector from " << argv[1] << "...\n";
    ReadEmbeddings(argv[1]);
    cerr << "VOCAB_SIZE: " << word_dict.size() << " \n";

    cerr << "Reading lexicons from " << argv[1] << "...\n";
    ReadLexicons(argv[2]);

    int tlc = 0;
    int ttoks = 0;
    ReadExamples(argv[3], training_examples, tlc, ttoks);

    //Freeze vocab after reading training corpus
    word_dict.Freeze();
    word_dict.SetUnk("<|unk|>"); //tzy
    kUNK = word_dict.Convert("<|unk|>"); //tzy
    VOCAB_SIZE = word_dict.size();

    int t_sent_c = 0;
    int t_sent_toks = 0;
    ReadExamples(argv[4], training_sent_examples, t_sent_c, t_sent_toks);

    unordered_map<int, int> word_counts;
    for(auto& example: training_sent_examples)
    {
        for(auto& word: example.real_word_ids)
        {
            word_counts[word]++;
        }
    }
    for(auto& word: word_counts)
    {
        if(word.second == 1)
        {
            cerr<< "Singleton: " << word_dict.Convert(word.first) << endl;
            singletons.insert(word.first);
        }
    }
    cerr<<"Num_of_Singleton: " << singletons.size() << endl;
    cerr<< "kUNK:"<< kUNK << endl;
    cerr << "VOCAB_SIZE: " << word_dict.size() << " \n";
    for(int i = 0; i< word_dict.size(); i++)
    {
        cerr<< word_dict.Convert(i) << endl;
    }
    exit(0);


    int dlc = 0;
    int dtoks = 0;
    ReadExamples(argv[5], dev_examples, dlc, dtoks);

    int telc = 0;
    int tetoks = 0;
    ReadExamples(argv[6], test_examples, telc, tetoks);
    //HANDLE UNK in the dev corpus: tzy
    int num_dev_unk = 0;
    for(unsigned sent_index = 0; sent_index < dev_examples.size(); sent_index++)
    {
        auto& example = dev_examples[ sent_index ];
        for(unsigned word_index = 0; word_index < example.words.size(); word_index++)
        {
            if(example.word_ids[word_index] == kUNK)
            {
                cerr<<"[DEV UNK]: " << example.words[word_index] << endl;
                num_dev_unk++;
            }
        }
    }

    int num_test_unk = 0;
    //HANDLE UNK in the dev corpus: tzy
    for(unsigned sent_index = 0; sent_index < test_examples.size(); sent_index++)
    {
        auto& example = test_examples[ sent_index ];
        for(unsigned word_index = 0; word_index < example.words.size(); word_index++)
        {
            if(example.word_ids[word_index] == kUNK )
                num_test_unk ++;
        }
    }
    cerr<<"NUM DEV UNK: " << num_dev_unk << endl;
    cerr<<"NUM TEST UNK: " << num_test_unk << endl;

    ostringstream os;
    os << "lstmclassifier"
       << '_' << INPUT_DIM
       << '_' << HIDDEN_DIM
       << '_' << LAYERS
       << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    float best = 0;

    Model model;
    bool use_momentum = true;
    Trainer* sgd = nullptr;
    if (use_momentum)
        sgd = new MomentumSGDTrainer(&model);
    else
        sgd = new SimpleSGDTrainer(&model);
    sgd->eta_decay = 0.1;

    LSTMClassifier lstmClassifier(model);


    unsigned report_every_i = min(1000, int(training_examples.size()));
    unsigned dev_report_every_i = 10;
    unsigned si = training_examples.size();
    vector<unsigned> order(training_examples.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int exceed_count = 0;
    int count = 0;
    while(count<=30) {
        Timer iteration("completed in");
        float loss = 0;
        unsigned ttags = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training_examples.size()) {
                si = 0;
                if (first) {
                    first = false;
                }
                else {
                    sgd->update_epoch();
                    if (1) {
                        float acc = 0.f;
                        cerr << "\n***DEV [epoch=" << (lines / (float)training_examples.size()) << "] ";
                        evaluate(dev_examples, lstmClassifier, acc);
                        if (acc > best) {
                            best = acc;
                            cerr<< "Exceed" << endl;
                            float tacc = 0;
                            evaluate(test_examples, lstmClassifier, tacc);
                            ofstream out(fname);
                            boost::archive::text_oarchive oa(out);
                            oa << model;
                            exceed_count ++;
                            /*if(exceed_count == 16)
                            {
                                delete sgd;
                                return 0;
                            }*/
                        }
                    }
                    //eval = false;
                }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                count++;
            }
            //cerr<<order[si] << endl;

            // build graph for this instance
            ComputationGraph cg;
            auto& sentx_y = training_examples[order[si]];
            for(int j = 0; j < (int)sentx_y.real_word_ids.size(); j++)
            {
                if(singletons.find(word_counts[sentx_y.real_word_ids[j]]) != singletons.end() && cnn::rand01() < punk)
                    sentx_y.word_ids[j] = kUNK;
                else
                    sentx_y.word_ids[j] = sentx_y.real_word_ids[j];
            }
            const int y = sentx_y.label;
            Expression y_pred = lstmClassifier.BuildGraph(sentx_y, cg);
            CrossEntropyLoss(y_pred, y);
            loss += as_scalar(cg.incremental_forward());
            cg.backward();
            sgd->update(1.0);
            ++si;
            ++lines;
            ++ttags;
        }
        sgd->status();
        cerr << " E = " << (loss / ttags) <<" "<<loss << "/"<<ttags<<" ";

        // show score on dev data?
        report++;
        continue;
        if ( report % dev_report_every_i == 1 ) {
            float acc = 0.f;
            cerr << "\n***DEV [epoch=" << (lines / (float)training_examples.size()) << "] ";
            evaluate(dev_examples, lstmClassifier, acc);
            if (acc > best) {
                best = acc;
                cerr<< "Exceed" << endl;
                float tacc = 0;
                evaluate(test_examples, lstmClassifier, tacc);
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
                exceed_count++;
                /*if(exceed_count == 16)
                {
                    delete sgd;
                    return 0;
                }*/
            }
        }
    }
    delete sgd;
}

