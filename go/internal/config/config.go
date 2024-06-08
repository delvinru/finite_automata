package config

import (
	"log/slog"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	N     int    `yaml:"n"`
	Phi   string `yaml:"phi"`
	Psi   string `yaml:"psi"`
	State string `yaml:"state"`
}

func Get() []map[string]Config {
	file, err := os.ReadFile("./configs/config.yaml")
	if err != nil {
		slog.Error("Error while reading yaml config file")
		os.Exit(1)
	}

	configs := []map[string]Config{}
	if err := yaml.Unmarshal(file, &configs); err != nil {
		slog.Error("Error while unmarshalling object")
		os.Exit(1)
	}

	return configs
}
