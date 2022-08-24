package es.urjc.etsii.grafo.autoconfig;

import es.urjc.etsii.grafo.autoconfig.exception.AlgorithmParsingException;
import org.apache.commons.lang3.ClassUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;

/**
 * Magic (based on reflection) util methods to create instances of algorithm components at runtime
 */
public class AlgorithmBuilderUtil {
    private static final Logger log = LoggerFactory.getLogger(AlgorithmBuilderUtil.class);

    private record UNKNOWNCLASS(){}

    /**
     * Build algorithm component reflectively
     * @param clazz Algorithm component class
     * @param args arguments for the constructor
     * @return instance if class built with the given params
     */
    public static Object build(Class<?> clazz, Map<String, Object> args){
        Map<String, Class<?>> argTypes = new HashMap<>();
        args.forEach((k, v) -> argTypes.put(k, v == null? UNKNOWNCLASS.class: v.getClass()));
        var constructor = findConstructor(clazz, argTypes);
        if(constructor == null){
            throw new AlgorithmParsingException(String.format("Failed to find constructor method in class %s for params %s, types %s", clazz.getSimpleName(), args, argTypes));
        }
        var params = new Object[args.size()];
        var cParams = constructor.getParameters();
        for (int i = 0; i < cParams.length; i++) {
            params[i] = args.get(cParams[i].getName());
        }
        try {
            return constructor.newInstance(params);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Find a constructor method in the target class that accepts the given combination of parameter (name, type) in any order.
     * Autoboxing and widening are allowed, example from int to double, or int to Integer.
     * @param clazz target class
     * @param args argument map
     * @return Constructor if found one that matches the given parameters, null if no constructor matches
     * @param <T> Constructor for class T
     */
    @SuppressWarnings("unchecked")
    public static <T> Constructor<T> findConstructor(Class<T> clazz, Map<String, Class<?>> args){
        var constructors = clazz.getConstructors();

        for(var c: constructors){
            boolean matches = doParamsMatch(c, args);
            if(matches){
                log.debug("Found matching constructor {} for args {}", c, args);
                return (Constructor<T>) c;
            }
        }
        log.debug("Failed to to found a matching constructor for class {} and args {}, detected constructors: {}", clazz.getSimpleName(), args, constructors);
        return null;
    }

    private static boolean doParamsMatch(Constructor<?> c, Map<String, Class<?>> params) {
        var cParams = c.getParameters();
        if(cParams.length != params.size()){
            log.debug("Constructor {} ignored, args size mismatch, |params|={}", c, params.size());
            return false;
        }

        for(var p: cParams){
            String cParamName = p.getName();
            var cParamClass = p.getType();
            if(!params.containsKey(cParamName)){
                log.debug("Constructor {} ignored, arg {} does not exist in map {}", c, cParamName, params);
                return false;
            }
            var parsedType = params.get(cParamName);
            if(parsedType == UNKNOWNCLASS.class && !cParamClass.isPrimitive()){
                // Null values do not have a known class, but can be used as any parameter as long as it is not a primitive type
                continue;
            }
            if(!ClassUtils.isAssignable(parsedType, cParamClass)){
                log.debug("Constructor {} ignored, arg {} with type {} is not assignable to {}, map {}", c, cParamName, parsedType, cParamClass, params);
                return false;
            }
        }
        return true;
    }

}
