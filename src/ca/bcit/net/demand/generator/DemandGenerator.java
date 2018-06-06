package ca.bcit.net.demand.generator;

import ca.bcit.io.YamlSerializable;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandStream;
import ca.bcit.utils.random.IrwinHallRandomVariable;
import ca.bcit.utils.random.RandomVariable;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public abstract class DemandGenerator<D extends Demand> implements DemandStream<D>, YamlSerializable {

	final RandomVariable<Boolean> reallocate;
	final RandomVariable<Boolean> allocateBackup;
	final RandomVariable<Integer> volume;
	RandomVariable<Integer> ttl;
	final RandomVariable<Float> squeezeRatio;
	final RandomVariable<Integer> cpu;
	final RandomVariable<Integer> memory;
	final RandomVariable<Integer> storage;

	int generatedDemandsCount;
	
	DemandGenerator(RandomVariable<Boolean> reallocate, RandomVariable<Boolean> allocateBackup, RandomVariable<Integer> volume, RandomVariable<Float> squeezeRatio, RandomVariable<Integer> cpu, RandomVariable<Integer> memory, RandomVariable<Integer> storage) {
		this.reallocate = reallocate;
		this.allocateBackup = allocateBackup;
		this.volume = volume;
		this.squeezeRatio = squeezeRatio;
		this.cpu = cpu;
		this.memory = memory;
		this.storage = storage;
	}
	
	public Random setSeed(long seed) {
		Random seedGenerator = new Random(seed);
		reallocate.setSeed(seedGenerator.nextLong());
		allocateBackup.setSeed(seedGenerator.nextLong());
		volume.setSeed(seedGenerator.nextLong());
		squeezeRatio.setSeed(seedGenerator.nextLong());
		ttl.setSeed(seedGenerator.nextLong());
		cpu.setSeed(seedGenerator.nextLong());
		memory.setSeed(seedGenerator.nextLong());
		storage.setSeed(seedGenerator.nextLong());
		generatedDemandsCount = 0;
		return seedGenerator;
	}
	
	public void setErlang(int erlang) {
		ttl = new IrwinHallRandomVariable.Integer(erlang - 50, erlang + 50, 10);
	}

	@Override
	public int getGeneratedDemandsCount() {
		return generatedDemandsCount;
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	DemandGenerator(Map map) {
		reallocate = (RandomVariable<Boolean>) map.get("reallocate");
		allocateBackup = (RandomVariable<Boolean>) map.get("allocateBackup");
		volume = (RandomVariable<Integer>) map.get("volume");
		squeezeRatio = (RandomVariable<Float>) map.get("squeezeRatio");
		ttl = (RandomVariable<Integer>) map.get("ttl");
		cpu = (RandomVariable<Integer>) map.get("cpu");
		memory = (RandomVariable<Integer>) map.get("memory");
		storage = (RandomVariable<Integer>) map.get("storage");
	}
	
	@Override
	public Map<String, Object> serialize() {
		Map<String, Object> map = new HashMap<>();
		
		map.put("reallocate", reallocate);
		map.put("allocateBackup", allocateBackup);
		map.put("volume", volume);
		map.put("squeezeRatio", squeezeRatio);
		map.put("ttl", ttl);
		map.put("cpu", cpu);
		map.put("memory", memory);
		map.put("storage", storage);
		return map;
	}
}