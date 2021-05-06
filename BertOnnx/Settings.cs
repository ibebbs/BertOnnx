namespace BertOnnx
{
    public class Settings
    {
        public string ModelPath { get; set; } = "./Assets/distilbert-base-cased-finetuned-conll03-english.onnx"; // "./Assets/model-optimized-quantized.onnx";

        public string ConfigPath { get; set; } = "./Assets/config.json";

        public string VocabPath { get; set; } = "./Assets/vocab.txt";

        public string[] ModelInput => new[] { "input_ids", "attention_mask" };

        public string[] ModelOutput => new[] { "output_0" };

        public int SequenceLength => 256;
    }
}
