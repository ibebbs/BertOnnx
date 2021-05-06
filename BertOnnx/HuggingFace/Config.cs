using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;

namespace HuggingFace
{

    public class Config
    {
        public static Config FromFile(string path)
        {
            using (var stream = File.OpenRead(path))
            {
                using (var textReader = new StreamReader(stream))
                {
                    using (var jsonReader = new JsonTextReader(textReader))
                    {
                        return JsonSerializer.CreateDefault().Deserialize<Config>(jsonReader);
                    }
                }
            }
        }

        public string _name_or_path { get; set; }
        public string activation { get; set; }
        public string[] architectures { get; set; }
        public float attention_dropout { get; set; }
        public int dim { get; set; }
        public float dropout { get; set; }
        public string finetuning_task { get; set; }
        public int hidden_dim { get; set; }
        public Dictionary<string, string> id2label { get; set; }
        public float initializer_range { get; set; }
        public Dictionary<string, string> label2id { get; set; }
        public int max_position_embeddings { get; set; }
        public string model_type { get; set; }
        public int n_heads { get; set; }
        public int n_layers { get; set; }
        public bool output_past { get; set; }
        public int pad_token_id { get; set; }
        public float qa_dropout { get; set; }
        public float seq_classif_dropout { get; set; }
        public bool sinusoidal_pos_embds { get; set; }
        public bool tie_weights_ { get; set; }
        public string transformers_version { get; set; }
        public int vocab_size { get; set; }
    }

}
